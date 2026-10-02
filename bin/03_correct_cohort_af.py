#!/usr/bin/env python3
"""
Recover the AggV3 cohort AC / AN / AF from the raw annotation parquets and join
them onto the processed annotations.

Why: the AF column in the raw parquets is the 1000 Genomes AF from the VEP CSQ
field (split-vep resolved %INFO/AF to the CSQ subfield), and AN was dropped in
process_vep. AC and AN in the raw parquets are correct, and the cohort AF is
exactly AC/AN, so it is recomputed here.

Steps (run individually or all at once):
  1. shards   all raw .../shard-S/subshard-*/annotations.parquet of one shard
              -> annotations_processed/shard-S/variant_metadata.parquet
              one row per unique (chrom, pos, ref, alt) with ac, an, af. No MAF filter.
  2. concat   all of the above -> annotations_processed/variant_metadata.parquet
  3. join     annotations_cadd_fill_na_maf001.parquet + variant_metadata.parquet
              on (chrom, pos, ref, alt)

Usage:
  python cohort_af.py all
  python cohort_af.py shards [--shard 60] [--overwrite]
  python cohort_af.py concat
  python cohort_af.py join [--annotations PATH] [--out PATH]
"""
import argparse
import glob
import logging
import os
import re
import sys
import time

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("cohort_af")

SESSION = os.path.expanduser("~/session_data")
PROC_DIR = os.path.join(SESSION, "Data_out/variant_annos/annotations_processed")

# Input batch folders, in PRIORITY order. A shard is read entirely from the
# first folder that contains it, never mixed across folders. The two
# annotations_pre_processing folders come first so they win over the original
# batches that also contain shard-20 / shard-90 (annotations_11_40,
# annotations_42_98). Pass --shards-dir (repeatable, same priority rule) to
# override.
DEFAULT_SHARDS_DIRS = [
    f"{SESSION}/Data_out/variant_annos/annotations_pre_processing/annotations_90",
    f"{SESSION}/Data_out/variant_annos/annotations_pre_processing/annotations_20",
    f"{SESSION}/filesystems/annotations_1_10",
    f"{SESSION}/filesystems/annotations_11_40",
    f"{SESSION}/filesystems/annotations_41",
    f"{SESSION}/filesystems/annotations_42_98",
]

# Per-shard output. NOTE: process_vep's OUT_VAR_METADATA uses this same path.
# Step 1 refuses to overwrite a file that is not one of ours (see shard_out_ok).
OUT_NAME = "variant_metadata.parquet"

CONCAT_OUT = os.path.join(PROC_DIR, "variant_metadata.parquet")
DEFAULT_ANNOTATIONS = os.path.join(PROC_DIR, "annotations_cadd_fill_na_maf001.parquet")

KEYS = ["chrom", "pos", "ref", "alt"]
SCHEMA = {"chrom": pl.Utf8, "pos": pl.Int64, "ref": pl.Utf8, "alt": pl.Utf8,
          "ac": pl.Int32, "an": pl.Int32, "af": pl.Float64, "id": pl.Utf8}

SHARD_RE = re.compile(r"^shard-(\d+)$")


# ── helpers ──────────────────────────────────────────────────────────────────

def af_expr() -> pl.Expr:
    """Cohort AF = AC/AN; null when AN is null or 0 (no called genotypes)."""
    return (pl.when(pl.col("an") > 0)
              .then(pl.col("ac").cast(pl.Float64) / pl.col("an"))
              .otherwise(None)
              .alias("af"))


def id_expr() -> pl.Expr:
    return pl.concat_str([pl.col(k).cast(pl.Utf8) for k in KEYS], separator=":").alias("id")


def is_ours(path: str) -> bool:
    """True if the parquet at path was written by this script (has ac/an/af)."""
    names = set(pl.scan_parquet(path).collect_schema().names())
    return {"ac", "an", "af"} <= names


def find_shards(shards_dirs: list[str]) -> dict[str, tuple[str, list[str], list[str]]]:
    """Map shard name -> (chosen batch folder, its subshard files, overridden folders).

    Each shard is taken from the first folder in shards_dirs that has it.
    """
    chosen: dict[str, tuple[str, list[str], list[str]]] = {}
    for d in shards_dirs:
        if not os.path.isdir(d):
            log.warning(f"Batch folder not found, skipped: {d}")
            continue
        for sd in sorted(glob.glob(os.path.join(d, "shard-*"))):
            shard = os.path.basename(sd)
            if not SHARD_RE.match(shard):
                continue
            files = sorted(glob.glob(os.path.join(sd, "subshard-*", "annotations.parquet")))
            if not files:
                continue
            if shard in chosen:
                chosen[shard][2].append(d)
            else:
                chosen[shard] = (d, files, [])
    if not chosen:
        sys.exit(f"No shard-*/subshard-*/annotations.parquet found under: {shards_dirs}")
    return chosen


# ── step 1: per-shard variant metadata ───────────────────────────────────────

def build_shard(files: list[str]) -> pl.DataFrame:
    """Unique variants with ac, an, af over all subshards of one shard."""
    for f in files:
        names = pl.scan_parquet(f).collect_schema().names()
        missing = [c for c in ("CHROM", "POS", "REF", "ALT", "AC", "AN") if c not in names]
        if missing:
            raise ValueError(f"{f}: missing columns {missing}")

    # The raw files have one row per consequence (per transcript, and per ANN
    # record), so each variant appears many times; a variant can also sit in
    # two neighbouring subshards. AC/AN are site-level, so every copy should
    # carry the same values: unique() over keys + ac + an leaves one row per
    # variant, and the check below catches it if that ever isn't true.
    lf = pl.concat([
        pl.scan_parquet(f).select(
            pl.col("CHROM").cast(pl.Utf8).alias("chrom"),
            pl.col("POS").cast(pl.Int64).alias("pos"),
            pl.col("REF").cast(pl.Utf8).alias("ref"),
            pl.col("ALT").cast(pl.Utf8).alias("alt"),
            pl.col("AC").cast(pl.Int32).alias("ac"),
            pl.col("AN").cast(pl.Int32).alias("an"),
        )
        for f in files
    ])
    df = lf.unique().collect(engine="streaming")

    n_keys = df.select(KEYS).n_unique()
    if n_keys != df.height:
        raise ValueError(f"{df.height - n_keys} variants have conflicting AC/AN "
                         "across their rows")
    return df.with_columns(af_expr(), id_expr()).sort(KEYS).select(list(SCHEMA))


def step_shards(shards_dirs: list[str], only: list[str] | None, overwrite: bool) -> None:
    shards = find_shards(shards_dirs)
    log.info("Batch folders (priority order):")
    for d in shards_dirs:
        log.info(f"  {d}")

    if only:
        wanted = {f"shard-{s}" for s in only}
        absent = wanted - set(shards)
        if absent:
            sys.exit(f"Requested shard(s) not found in any batch folder: {sorted(absent)}")
        shards = {k: v for k, v in shards.items() if k in wanted}

    order = sorted(shards, key=lambda s: int(s.split("-")[1]))
    log.info(f"{len(order)} shard(s) to process")
    for s in order:
        d, files, overridden = shards[s]
        extra = f"; ignoring the copy in {[os.path.basename(o) for o in overridden]}" if overridden else ""
        log.info(f"  {s}: {len(files)} subshards from {os.path.basename(d)}{extra}")

    done = skipped = 0
    failed = []
    t0 = time.perf_counter()
    for i, s in enumerate(order, 1):
        _, files, _ = shards[s]
        out = os.path.join(PROC_DIR, s, OUT_NAME)
        if os.path.exists(out):
            if not is_ours(out):
                failed.append((s, f"{out} exists and is not a cohort-AF file (no ac/an/af; "
                                  "probably process_vep's OUT_VAR_METADATA). Move it aside "
                                  "first; it is never overwritten by this script."))
                log.error(f"[{i}/{len(order)}] {s}: {failed[-1][1]}")
                continue
            if not overwrite:
                skipped += 1
                continue
        try:
            t = time.perf_counter()
            df = build_shard(files)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            df.write_parquet(out, compression="zstd")
            done += 1
            log.info(f"[{i}/{len(order)}] {s}: {df.height:,} variants from "
                     f"{len(files)} subshards ({time.perf_counter() - t:.0f}s)")
        except Exception as e:  # keep going; report all failures at the end
            failed.append((s, str(e)))
            log.error(f"[{i}/{len(order)}] {s}: {e}")

    log.info(f"Step 1: {done} written, {skipped} already present (skipped), "
             f"{len(failed)} failed, {time.perf_counter() - t0:.0f}s")
    if failed:
        sys.exit("Failed shards:\n" + "\n".join(f"  {s}: {e}" for s, e in failed))


# ── step 2: concat ───────────────────────────────────────────────────────────

def step_concat() -> None:
    files = sorted(glob.glob(os.path.join(PROC_DIR, "shard-*", OUT_NAME)),
                   key=lambda p: int(p.split(os.sep)[-2].split("-")[1]))
    if not files:
        sys.exit(f"No shard-*/{OUT_NAME} under {PROC_DIR}; run step 1.")
    foreign = [f for f in files if not is_ours(f)]
    if foreign:
        sys.exit("These are not cohort-AF files (no ac/an/af), so they cannot be "
                 "concatenated; rerun step 1 for those shards:\n  " + "\n  ".join(foreign))
    log.info(f"Concatenating {len(files)} shard files")

    lf = pl.scan_parquet(files, schema=SCHEMA)

    # Shards are disjoint regions, so duplicates across shards should be rare.
    # Identical copies are collapsed; copies that disagree on AC/AN are an error.
    dedup = lf.unique()
    conflicts = (dedup.group_by(KEYS).len().filter(pl.col("len") > 1)
                 .select(pl.len()).collect(engine="streaming").item())
    if conflicts:
        sys.exit(f"{conflicts:,} variants appear in several shards with different "
                 "AC/AN; inspect before concatenating.")

    total = lf.select(pl.len()).collect(engine="streaming").item()
    dedup.sink_parquet(CONCAT_OUT, compression="zstd", engine="streaming")
    n = pl.scan_parquet(CONCAT_OUT).select(pl.len()).collect().item()
    log.info(f"Step 2: {n:,} unique variants ({total - n:,} cross-shard duplicates "
             f"removed) -> {CONCAT_OUT}")


# ── step 3: join ─────────────────────────────────────────────────────────────

def step_join(annotations: str, out: str) -> None:
    if os.path.abspath(annotations) == os.path.abspath(out):
        sys.exit("--out must differ from --annotations (refusing to overwrite the input).")

    annos = pl.scan_parquet(annotations)
    names = annos.collect_schema().names()

    # Join keys: use chrom/pos/ref/alt if the file has them, otherwise derive
    # them from id (process_vep builds id as chrom:pos:ref:alt).
    if all(k in names for k in KEYS):
        key_src = "columns"
        annos = annos.with_columns(
            pl.col("chrom").cast(pl.Utf8).alias("_chrom"),
            pl.col("pos").cast(pl.Int64).alias("_pos"),
            pl.col("ref").cast(pl.Utf8).alias("_ref"),
            pl.col("alt").cast(pl.Utf8).alias("_alt"),
        )
    elif "id" in names:
        key_src = "id"
        parts = pl.col("id").str.split_exact(":", 3)
        annos = annos.with_columns(
            parts.struct.field("field_0").alias("_chrom"),
            parts.struct.field("field_1").cast(pl.Int64).alias("_pos"),
            parts.struct.field("field_2").alias("_ref"),
            parts.struct.field("field_3").alias("_alt"),
        )
    else:
        sys.exit(f"{annotations}: neither chrom/pos/ref/alt nor id present.")
    log.info(f"Join keys taken from {key_src}")

    # Replace stale columns rather than ending up with ac / ac_right etc.:
    #   ac               identical to the new one, but replaced for consistency
    #   af, an           not expected, dropped if present
    #   maf_cohort(_is_na) were the 1000 Genomes AF with nulls filled; rewritten
    #                    from the cohort AF so downstream configs keep working
    stale = [c for c in ("ac", "an", "af", "maf_cohort", "maf_cohort_is_na") if c in names]
    if stale:
        log.info(f"Replacing existing columns: {stale}")
    had_maf_cohort = "maf_cohort" in names
    had_is_na = "maf_cohort_is_na" in names
    maf_dtype = annos.collect_schema()["maf_cohort"] if had_maf_cohort else pl.Float32

    vm = (pl.scan_parquet(CONCAT_OUT)
            .select(pl.col("chrom").alias("_chrom"), pl.col("pos").alias("_pos"),
                    pl.col("ref").alias("_ref"), pl.col("alt").alias("_alt"),
                    "ac", "an", "af"))

    joined = (annos.drop(stale)
                   .join(vm, on=["_chrom", "_pos", "_ref", "_alt"], how="left")
                   .drop(["_chrom", "_pos", "_ref", "_alt"]))
    if had_maf_cohort:
        joined = joined.with_columns(pl.col("af").cast(maf_dtype).alias("maf_cohort"))
    if had_is_na:
        joined = joined.with_columns(
            pl.col("af").is_null().cast(pl.Int8).alias("maf_cohort_is_na"))

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    joined.sink_parquet(out, compression="zstd", engine="streaming")

    # Report: row count must not change (metadata is unique per variant), and
    # any variant without a match ends up with null ac/an/af.
    res = pl.scan_parquet(out)
    n_in = pl.scan_parquet(annotations).select(pl.len()).collect().item()
    stats = res.select(
        pl.len().alias("rows"),
        pl.col("ac").is_null().sum().alias("no_match"),
        pl.col("af").is_null().sum().alias("af_null"),
    ).collect().row(0, named=True)
    if stats["rows"] != n_in:
        sys.exit(f"Row count changed in join: {n_in:,} -> {stats['rows']:,}. "
                 "variant_metadata.parquet is not unique per variant.")
    log.info(f"Step 3: {stats['rows']:,} rows, {stats['no_match']:,} without a "
             f"metadata match, {stats['af_null']:,} with null af -> {out}")
    if stats["no_match"]:
        log.warning("Some annotation rows found no cohort AC/AN. Check that step 1 "
                    "covered every shard in the annotations file.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["shards", "concat", "join", "all"])
    p.add_argument("--shards-dir", action="append", default=None,
                   help="Batch folder with shard-*/subshard-*/annotations.parquet. "
                        "Repeatable; earlier folders take priority per shard. "
                        "Default: the six folders in DEFAULT_SHARDS_DIRS.")
    p.add_argument("--shard", action="append", default=None,
                   help="Step 1: only this shard number (repeatable), e.g. --shard 60.")
    p.add_argument("--overwrite", action="store_true",
                   help="Step 1: rebuild shard files this script already wrote.")
    p.add_argument("--annotations", default=DEFAULT_ANNOTATIONS,
                   help="Step 3 input (default: %(default)s).")
    p.add_argument("--out", default=None,
                   help="Step 3 output (default: <annotations>_cohort_af.parquet).")
    a = p.parse_args()

    shards_dirs = [os.path.expanduser(d) for d in (a.shards_dir or DEFAULT_SHARDS_DIRS)]
    out = a.out or a.annotations.replace(".parquet", "_cohort_af.parquet")

    if a.step in ("shards", "all"):
        step_shards(shards_dirs, a.shard, a.overwrite)
    if a.step in ("concat", "all"):
        step_concat()
    if a.step in ("join", "all"):
        step_join(a.annotations, out)


if __name__ == "__main__":
    main()