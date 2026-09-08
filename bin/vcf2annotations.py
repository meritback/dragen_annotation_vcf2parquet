#!/usr/bin/env python3
import polars as pl
import subprocess
import click
import tempfile
import re
import os
import sys
from polars.exceptions import NoDataError

# VEP-annotated msVCF -> long-format annotation parquet.
#
# paths:
# base_path  = "/home/vscode/session_data/filesystems/"
# annotated  = functional-annotation_2025-12-24/shard-{S}/subshard-{U}/dragen.gel.annotated.vcf.gz
#
# One row per variant per VEP consequence (i.e. per transcript). Site-level:
# there is no sample dimension here, so these files are small compared to the
# genotype parquets despite the ~200 columns.

BCFTOOLS_PATH = "bcftools"

# Same dtypes as 1_vcf2parquet.py, and for the same reason: these parquets are
# joined to merged.parquet on CHROM/POS/REF/ALT, and polars will error or
# silently upcast if POS is Int64 in one frame and Int32 in the other.
CHROM_ENUM = pl.Enum([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"])
POS_DTYPE = pl.Int64

# Fixed columns, ahead of the expanded CSQ block. Uppercase, unlike the CSQ
# subfields below, because these four are the join keys and must match the
# genotype parquets exactly.
#
# %ID is deliberately absent: tsv_to_parquet appends its own ID column
# (CHROM:POS:REF:ALT) and the names would collide. rsIDs are in the CSQ
# Existing_variation subfield anyway.
FIXED_COLS = ["CHROM", "POS", "REF", "ALT", "QUAL", "FILTER", "AC", "AN", "AF"]
FIXED_FMT = ("%CHROM\t%POS\t%REF\t%ALT\t%QUAL\t%FILTER"
             "\t%INFO/AC\t%INFO/AN\t%INFO/AF")

# The annotation field NOT being expanded, carried whole and unparsed as one
# string column. split-vep expands one field per invocation, and CSQ (VEP) is
# by far the richer of the two, so ANN (SnpEff) rides along as a raw block
# repeated on each CSQ row for that variant. Lossless; split it downstream if
# you need it. ZSTD compresses the repetition to almost nothing.
CARRY = {"CSQ": "ANN", "ANN": "CSQ"}

# Numeric casts. Everything is read as Utf8 first, so an unexpected value
# becomes null instead of killing the job halfway through a shard.
#
# Only the fixed columns are cast. The CSQ subfields stay Utf8 on purpose:
# many are multi-valued ("&"-separated per allele or per prediction), VEP
# writes "" and "." and "-" for missing in different fields, and a strict=False
# cast would silently null every one of those. Cast at query time with
# try_cast/strict=False, where you can see what you lost.
CASTS = {
    "POS": POS_DTYPE,
    "QUAL": pl.Float64,
    "AC": pl.Int32,
    "AN": pl.Int32,
    "AF": pl.Float64,
}


def bcf_to_tsv(bcf_file: str, output_file, format_str: str,
               annotation_field: str = "CSQ",
               view_exclude: str | None = None,
               region: str | None = None) -> None:
    """Expand the annotation field to TSV, one row per consequence.

    -a pins which INFO field is parsed. Without it split-vep auto-detects in
    the order CSQ, BCSQ, ANN and warns when more than one is present -- and
    this script calls bcftools twice, once for the header and once for the
    data. If those two calls ever resolved to different fields the column
    names and the values would silently disagree.

    -d  one output row per consequence entry
    -x  skip sites with no annotation at all
    -u  print '.' for tags missing from this file's header rather than failing,
        so a subshard missing INFO/AC does not take down the task
    -A tab  expand every subfield into its own column, in header order
    """
    region_arg = f"--regions '{region}' " if region else ""

    pre = ""
    if view_exclude or region_arg:
        pre = f"{BCFTOOLS_PATH} view {region_arg}"
        if view_exclude:
            pre += f"--exclude '{view_exclude}' "
        pre += f"-Ou \"{bcf_file}\" | "
        src = "-"
    else:
        src = f'"{bcf_file}"'

    cmd = (f"{pre}{BCFTOOLS_PATH} +split-vep -a {annotation_field} {src} "
           f"-d -x -u -A tab --format '{format_str}'")

    print(f"Executing: {cmd}")
    proc = subprocess.run(cmd, shell=True, stdout=output_file)
    if proc.returncode != 0:
        # Negative returncode means the child died on a signal. Re-encode as the
        # shell convention 128+N so Nextflow's errorStrategy sees the real cause:
        # a SIGKILLed bcftools becomes 137, which is in the retry list, instead of
        # being flattened to 1 by CalledProcessError and finishing the run.
        rc = 128 - proc.returncode if proc.returncode < 0 else proc.returncode
        sys.exit(rc)


def csq_columns(bcf_file: str, annotation_field: str = "CSQ") -> list:
    """Column names for the expanded annotation block, in header order.

    `split-vep -l` parses the Description= of the INFO field into its subfield
    names. Sanitised to lowercase snake_case, because VEP field names are not
    valid identifiers ('cDNA_position', 'MAX_AF_POPS', 'CDS.pos / CDS.length')
    and downstream SQL against them would need quoting everywhere.

    Positional alignment with the data pass is the whole game here: -A tab
    emits subfields in exactly this order, so the Nth name must describe the
    Nth column. Nothing downstream re-checks that, which is why -a is pinned
    in both calls.
    """
    out = subprocess.run(
        [BCFTOOLS_PATH, "+split-vep", "-a", annotation_field, "-l", bcf_file],
        check=True, capture_output=True, text=True,
    ).stdout

    names = []
    # Reserve the fixed names and the ID column tsv_to_parquet appends, so a
    # CSQ subfield can never shadow a join key.
    seen = ({c.lower() for c in FIXED_COLS} | set(FIXED_COLS)
            | {"ID", "id"} | {k.lower() for k in CARRY} | set(CARRY))
    for line in out.splitlines():
        # "<index>\t<name>"; take the name.
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        raw = parts[1].strip()
        if not raw:
            continue
        s = re.sub(r"[^0-9a-zA-Z]+", "_", raw).strip("_").lower() or "field"
        base, i = s, 1
        # Distinct VEP fields can sanitise to the same identifier
        # ('CDS.pos / CDS.length' vs 'CDS_pos_CDS_length'). Suffix rather than
        # drop: losing a column silently would shift every column after it.
        while s in seen:
            i += 1
            s = f"{base}_{i}"
        seen.add(s)
        names.append(s)

    if not names:
        sys.exit(f"no {annotation_field} subfields in the header of {bcf_file}")
    return names


def write_empty_parquet(output_file: str, column_names: list):
    """Zero-row parquet with the schema a populated one would have.

    Built from the same column list, in the same order, with ID appended last
    -- otherwise concatenating subshards later fails on schema mismatch.
    """
    schema = {c: CASTS.get(c, pl.Utf8) for c in column_names}
    schema["CHROM"] = CHROM_ENUM
    schema["ID"] = pl.Utf8
    pl.DataFrame(schema=schema).write_parquet(output_file, compression="zstd")


def tsv_to_parquet(tsv_file: str, output_file: str, column_names: list):
    """Convert the split-vep TSV to Parquet.

    Read everything as Utf8, cast afterwards with strict=False. No sort: the
    bcftools output is already in coordinate order, so the parquet row-group
    statistics are useful for predicate pushdown without an extra pass.
    """
    cols = set(column_names)
    try:
        df = pl.scan_csv(
            tsv_file,
            separator="\t",
            has_header=False,
            new_columns=column_names,
            schema={c: pl.Utf8 for c in column_names},
            null_values=[".", ""],
            # This is bcftools TSV, not CSV. Left on, a stray double quote in
            # an HGVS or domain string would swallow tabs and newlines until
            # the next one and misalign every column after it.
            quote_char=None,
        )

        df = df.with_columns(
            [pl.col(c).cast(t, strict=False) for c, t in CASTS.items() if c in cols]
        )
        df = df.with_columns(pl.col("CHROM").cast(CHROM_ENUM, strict=False))
        df = df.with_columns(
            pl.concat_str(
                [pl.col("CHROM").cast(pl.Utf8), pl.col("POS").cast(pl.Utf8),
                 pl.col("REF"), pl.col("ALT")],
                separator=":",
            ).alias("ID")
        )

        df.sink_parquet(output_file, engine="streaming", compression="zstd")

    except NoDataError:
        print("Warning: No variants found. Creating empty Parquet file.")
        write_empty_parquet(output_file, column_names)


def check_bcftools():
    if subprocess.call(f"type {BCFTOOLS_PATH}", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        sys.exit(f"Error: Could not find bcftools at '{BCFTOOLS_PATH}'.")
    # +split-vep is a plugin: bioconda ships it, but BCFTOOLS_PLUGINS has to
    # point at it in some container layouts. Fail here with a clear message
    # rather than inside a shell pipeline.
    rc = subprocess.call(f"{BCFTOOLS_PATH} +split-vep -h", shell=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc not in (0, 1):
        sys.exit("Error: bcftools +split-vep plugin not available. "
                 "Set BCFTOOLS_PLUGINS to the plugin directory.")


@click.group()
def cli():
    """VEP-annotated VCF to Parquet converter."""
    pass


@cli.command("convert_annotations")
@click.argument("bcf_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--columns-file", type=click.Path(), default=None,
              help="Write the resolved column list here, one per line. The "
                   "pipeline collects these and checks every subshard agrees: "
                   "parquets with different schemas will not concatenate.")
@click.option("--annotation-field", default="CSQ", show_default=True,
              help="INFO field to expand. CSQ is VEP; ANN is SnpEff and much "
                   "thinner. Whichever is not chosen is carried as a raw "
                   "string column if it is in FIXED_FMT.")
@click.option("--region", default=None, help="e.g. chr1:1000000-1100000 (testing)")
@click.option("--max-af", type=float, default=None,
              help="Exclude sites with INFO/AF above this. At AggV3's cohort "
                   "size this removes only a few percent of sites -- the site "
                   "frequency spectrum is dominated by singletons -- so "
                   "filtering at query time is usually the better trade.")
@click.option("--max-ac", type=int, default=None)
def convert_annotations(bcf_file: str, output_file: str, columns_file: str,
                        annotation_field: str, region: str,
                        max_af: float, max_ac: int):
    """Annotated VCF -> parquet. One row per variant per consequence."""
    check_bcftools()
    click.echo(f"Processing {bcf_file}...")

    carry = CARRY.get(annotation_field)
    csq = csq_columns(bcf_file, annotation_field)

    column_names = list(FIXED_COLS)
    format_str = FIXED_FMT
    if carry:
        column_names.append(carry)
        format_str += f"\t%INFO/{carry}"
    column_names += csq
    format_str += f"\t%{annotation_field}\n"

    click.echo(f"{annotation_field} subfields: {len(csq)}  "
               f"total columns: {len(column_names)}")

    if columns_file:
        with open(columns_file, "w") as fh:
            fh.write("\n".join(column_names) + "\n")

    # ALT="*" is the spanning-deletion placeholder, not a real allele, and it
    # carries no meaningful consequence.
    excl = ['ALT="*"']
    if max_af is not None:
        excl.append(f"INFO/AF>{max_af}")
    if max_ac is not None:
        excl.append(f"INFO/AC>{max_ac}")
    site_exclude = " || ".join(excl)

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as workdir:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=workdir) as tmp:
            bcf_to_tsv(bcf_file, tmp, format_str=format_str,
                       annotation_field=annotation_field,
                       view_exclude=site_exclude,
                       region=region)
            tmp.flush()
            tsv_to_parquet(tmp.name, output_file, column_names)

    click.echo(f"Success! Parquet file created: {output_file}")


@cli.command("list_fields")
@click.argument("bcf_file", type=click.Path(exists=True))
@click.option("--annotation-field", default="CSQ", show_default=True)
def list_fields(bcf_file: str, annotation_field: str):
    """Print the resolved column list for one VCF, without converting.

    Worth running on a couple of subshards before a full submission: it is the
    cheapest way to confirm the header is what you think it is.
    """
    check_bcftools()
    carry = CARRY.get(annotation_field)
    cols = list(FIXED_COLS) + ([carry] if carry else [])
    for c in cols + csq_columns(bcf_file, annotation_field):
        click.echo(c)


if __name__ == "__main__":
    cli()
