#!/usr/bin/env python3
import polars as pl
import subprocess
import click
import tempfile
import re
import os
import sys
from polars.exceptions import NoDataError

# VEP/SnpEff-annotated msVCF -> long-format annotation parquet, one file per
# annotation field.
#
# paths:
# base_path  = "/home/vscode/session_data/filesystems/"
# annotated  = functional-annotation_2025-12-24/shard-{S}/subshard-{U}/dragen.gel.annotated.vcf.gz
#
# One row per variant per consequence (i.e. per transcript), per field:
#
#   annotations_CSQ.parquet   VEP,    ~200 subfield columns, ~40 rows/variant
#   annotations_ANN.parquet   SnpEff,  16 subfield columns,  ~2 rows/variant
#
# Both carry the same fixed key columns and an ID (CHROM:POS:REF:ALT), so they
# join to each other and to the genotype pipeline's merged.parquet.
#
# Why two files rather than one: split-vep expands exactly one field per
# invocation, and -d multiplies rows by that field's entry count. Carrying the
# other field along as a raw string in the same pass reprints its entire block
# on every row -- measured at ~40 CSQ consequences per variant on this release,
# i.e. 39 redundant copies of ANN through the TSV, the pipe and the parser.
# The two fields are also per-transcript lists of different lengths with no
# correct row-wise pairing, so there is no single table that holds both
# without either duplication or invented correspondences.

BCFTOOLS_PATH = "bcftools"

# Same dtypes as 1_vcf2parquet.py, and for the same reason: these parquets are
# joined to merged.parquet on CHROM/POS/REF/ALT, and polars will error or
# silently upcast if POS is Int64 in one frame and Int32 in the other.
CHROM_ENUM = pl.Enum([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"])
POS_DTYPE = pl.Int64

# Fixed columns, ahead of the expanded subfield block. Uppercase, unlike the
# subfields below, because the first four are the join keys and must match the
# genotype parquets exactly. Written identically into every field's file.
#
# %ID is deliberately absent: tsv_to_parquet appends its own ID column
# (CHROM:POS:REF:ALT) and the names would collide. rsIDs are in the CSQ
# Existing_variation subfield anyway.
FIXED_COLS = ["CHROM", "POS", "REF", "ALT", "QUAL", "FILTER", "AC", "AN", "AF"]
FIXED_FMT = ("%CHROM\t%POS\t%REF\t%ALT\t%QUAL\t%FILTER"
             "\t%INFO/AC\t%INFO/AN\t%INFO/AF")

# Names a sanitised subfield must never take, or it would shadow a join key.
RESERVED = ({c.lower() for c in FIXED_COLS} | set(FIXED_COLS)
            | {"ID", "id"} | {"csq", "ann", "bcsq", "CSQ", "ANN", "BCSQ"})

# Numeric casts. Everything is read as Utf8 first, so an unexpected value
# becomes null instead of killing the job halfway through a shard.
#
# Only the fixed columns are cast. The subfields stay Utf8 on purpose: many
# are multi-valued ("&"-separated per allele or per prediction), VEP writes
# "" and "." and "-" for missing in different fields, and a strict=False cast
# would silently null every one of those. Cast at query time with
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
    """Expand one annotation field to TSV, one row per consequence.

    -a pins which INFO field is parsed. Without it split-vep auto-detects in
    the order CSQ, BCSQ, ANN and warns when more than one is present -- and
    each field is processed by two bcftools calls, once for the header and
    once for the data. If those two resolved to different fields the column
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
    """Column names for one expanded annotation block, in header order.

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
    seen = set(RESERVED)
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
        # Distinct fields can sanitise to the same identifier ('CDS.pos /
        # CDS.length' vs 'CDS_pos_CDS_length'). Suffix rather than drop:
        # losing a column silently would shift every column after it.
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


def convert_one_field(bcf_file: str, field: str, out_parquet: str,
                      columns_file: str | None, site_exclude: str | None,
                      region: str | None, workdir: str) -> int:
    """One field -> one parquet. Returns the number of columns written.

    Each field is a self-contained pass: its own header lookup, its own
    split-vep invocation, its own TSV. Nothing from the other field is in
    the format string, which is the point of the split.
    """
    subfields = csq_columns(bcf_file, field)
    column_names = list(FIXED_COLS) + subfields
    format_str = FIXED_FMT + f"\t%{field}\n"

    click.echo(f"[{field}] subfields: {len(subfields)}  "
               f"total columns: {len(column_names)} (+ID)")

    if columns_file:
        with open(columns_file, "w") as fh:
            fh.write("\n".join(column_names) + "\n")

    with tempfile.NamedTemporaryFile(mode="w+b", dir=workdir) as tmp:
        bcf_to_tsv(bcf_file, tmp, format_str=format_str,
                   annotation_field=field,
                   view_exclude=site_exclude,
                   region=region)
        tmp.flush()
        tsv_to_parquet(tmp.name, out_parquet, column_names)

    click.echo(f"[{field}] wrote {out_parquet}")
    return len(column_names)


def check_bcftools():
    if subprocess.call(f"type {BCFTOOLS_PATH}", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        sys.exit(f"Error: Could not find bcftools at '{BCFTOOLS_PATH}'.")
    p = subprocess.run([BCFTOOLS_PATH, "plugin", "-l"],
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    if "split-vep" not in out:
        sys.exit("Error: bcftools +split-vep plugin not available.\n"
                 f"BCFTOOLS_PLUGINS={os.environ.get('BCFTOOLS_PLUGINS', '<unset>')}\n"
                 f"{out[:500]}")


def parse_fields(fields: str) -> list:
    out = []
    for f in fields.split(","):
        f = f.strip().upper()
        if f and f not in out:
            out.append(f)
    if not out:
        sys.exit("--fields resolved to nothing")
    return out


@click.group()
def cli():
    """Annotated VCF to Parquet converter."""
    pass


@cli.command("convert_annotations")
@click.argument("bcf_file", type=click.Path(exists=True))
@click.argument("output_prefix", type=click.Path())
@click.option("--fields", default="CSQ,ANN", show_default=True,
              help="Comma-separated INFO fields to expand, one parquet each. "
                   "CSQ is VEP (~200 subfields); ANN is SnpEff (16). Each is "
                   "expanded in its own split-vep pass, so neither is "
                   "duplicated across the other's rows.")
@click.option("--columns-prefix", type=click.Path(), default=None,
              help="Write each field's resolved column list to "
                   "<prefix>_<FIELD>.txt, one name per line. The pipeline "
                   "collects these and checks every subshard agrees: parquets "
                   "with different schemas will not concatenate.")
@click.option("--region", default=None, help="e.g. chr1:1000000-1100000 (testing)")
@click.option("--max-af", type=float, default=None,
              help="Exclude sites with INFO/AF above this. At AggV3's cohort "
                   "size this removes only a few percent of sites -- the site "
                   "frequency spectrum is dominated by singletons -- so "
                   "filtering at query time is usually the better trade.")
@click.option("--max-ac", type=int, default=None)
def convert_annotations(bcf_file: str, output_prefix: str, fields: str,
                        columns_prefix: str, region: str,
                        max_af: float, max_ac: int):
    """Annotated VCF -> one parquet per annotation field.

    Writes <output_prefix>_<FIELD>.parquet for each field, e.g.
    annotations_CSQ.parquet and annotations_ANN.parquet.
    """
    check_bcftools()
    field_list = parse_fields(fields)
    click.echo(f"Processing {bcf_file}  fields: {', '.join(field_list)}")

    # ALT="*" is the spanning-deletion placeholder, not a real allele, and it
    # carries no meaningful consequence. Applied identically to every field's
    # pass, so the row sets stay comparable and the IDs join.
    excl = ['ALT="*"']
    if max_af is not None:
        excl.append(f"INFO/AF>{max_af}")
    if max_ac is not None:
        excl.append(f"INFO/AC>{max_ac}")
    site_exclude = " || ".join(excl)

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as workdir:
        for field in field_list:
            convert_one_field(
                bcf_file, field,
                out_parquet=f"{output_prefix}_{field}.parquet",
                columns_file=(f"{columns_prefix}_{field}.txt"
                              if columns_prefix else None),
                site_exclude=site_exclude,
                region=region,
                workdir=workdir,
            )

    click.echo("Success! " + "  ".join(
        f"{output_prefix}_{f}.parquet" for f in field_list))


@cli.command("list_fields")
@click.argument("bcf_file", type=click.Path(exists=True))
@click.option("--fields", default="CSQ,ANN", show_default=True)
def list_fields(bcf_file: str, fields: str):
    """Print the resolved column list per field, without converting.

    Worth running on a couple of subshards before a full submission: it is the
    cheapest way to confirm the header is what you think it is, and the only
    cheap way to find out whether split-vep can parse this release's ANN
    header at all.
    """
    check_bcftools()
    for field in parse_fields(fields):
        click.echo(f"# {field}")
        for c in list(FIXED_COLS) + csq_columns(bcf_file, field):
            click.echo(c)
        click.echo("ID")
        click.echo("")


if __name__ == "__main__":
    cli()