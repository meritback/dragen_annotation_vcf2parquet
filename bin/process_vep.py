import gc
import logging
import os
import re
import sys
import yaml

import numpy as np
import pandas as pd
import polars as pl
import pyranges as pr
from pathlib import Path
from tqdm import tqdm
import argparse
import glob

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="log.txt",
    filemode="w",
)

SPLICE = ["ag", "al", "dg", "dl"]
# Fixed casts only; gnomAD columns are added per frame inside the function
BASE_CASTS = {
    "POS": pl.Int32, "QUAL": pl.Float32, "AC": pl.Int32, "AN": pl.Int32, "AF": pl.Float32,
    "distance": pl.Int32,
    # "strand": pl.Int8,
    "tsl": pl.Int8,
    "existing_inframe_oorfs": pl.Int16,
    "existing_outofframe_oorfs": pl.Int16,
    "existing_uorfs": pl.Int16,
    **{c: pl.Float32 for c in ["af_2", "afr_af", "amr_af", "eas_af", "eur_af", "sas_af",
                               "max_af", "cadd_phred", "cadd_raw", "revel"]},
    **{f"spliceai_pred_ds_{s}": pl.Float32 for s in SPLICE},
    **{f"spliceai_pred_dp_{s}": pl.Int16 for s in SPLICE},
}


ANN_COLUMNS = [
    "ANN",
    "annotation",
    "annotation_impact",
    "gene_name",
    "gene_id",
    "feature_type",
    "feature_id",
    "transcript_biotype",
    "rank",
    "hgvs_c",
    "hgvs_p",
    "cdna_pos_cdna_length",
    "cds_pos_cds_length",
    "aa_pos_aa_length",
    "errors_warnings_info",
]
VEP_TO_DROP = ["mechpredict_pdn",
    "mechpredict_pgof",
    "mechpredict_plof",
    "mechpredict_prediction"]


# https://github.com/HolEv/deeprvat_wgs/blob/main/scripts/annotation/annotation_functions.py


# ── BLOSUM62 substitution matrix ───────────────────────────────────────────
# curl -s "https://ftp.ncbi.nlm.nih.gov/repository/blocks/unix/blosum/BLOSUM/blosum62.sij"
def _load_blosum62_sij(path: str) -> pl.DataFrame:
    """Parse an NCBI .sij lower-triangle matrix file into a symmetrised Polars DataFrame."""
    aa_order = []
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            tokens = line.split()
            if not tokens:
                continue
            # Header line: all tokens are single uppercase letters
            if all(len(t) == 1 and t.isupper() for t in tokens):
                aa_order = tokens
            else:
                rows.append([float(v) for v in tokens])
    n = len(aa_order)
    assert len(rows) == n, f"Expected {n} data rows, got {len(rows)}"
    pairs_ref, pairs_alt, scores = [], [], []
    for i, aa1 in enumerate(aa_order):
        for j, aa2 in enumerate(aa_order):
            val = rows[max(i, j)][min(i, j)]
            pairs_ref.append(aa1)
            pairs_alt.append(aa2)
            scores.append(val)
    return pl.DataFrame(
        {"ref_aa": pairs_ref, "alt_aa": pairs_alt, "blosum62": scores},
        schema={"ref_aa": pl.Utf8, "alt_aa": pl.Utf8, "blosum62": pl.Float32},
    )


# ── Low-level helpers ──────────────────────────────────────────────────────
# SHUBHANKAR
 
def _load_gtf_polars(gtf_path: str) -> pl.DataFrame:
    """Read a (possibly gzipped) Gencode GTF into a polars DataFrame."""
    return pl.read_csv(
        gtf_path,
        separator="\t",
        comment_prefix="#",
        has_header=False,
        new_columns=[
            "chrom", "source", "feature", "start", "end",
            "score", "strand", "frame", "attributes",
        ],
        schema_overrides={
            "chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64,
            "feature": pl.Utf8, "strand": pl.Utf8, "attributes": pl.Utf8,
        },
        ignore_errors=True,
    )
 
 
def _gtf_gene_features(gtf: pl.DataFrame) -> pl.DataFrame:
    """Per-gene TSS, length and name from GTF 'gene' rows (protein-coding only).
 
    Returns columns: region, gene_name, gene_length, _tss, _gene_strand.
    """
    return (
        gtf.filter(pl.col("feature") == "gene")
        .with_columns(
            region=pl.col("attributes").str.extract(r'gene_id "([^"]+)"')
                  .str.split(".").list.first(),
            gene_name=pl.col("attributes").str.extract(r'gene_name "([^"]+)"'),
            gene_type=pl.col("attributes")
                  .str.extract(r'gene_(?:type|biotype) "([^"]+)"'),
        )
        .filter(pl.col("gene_type") == "protein_coding")
        .with_columns(
            gene_length=pl.col("end") - pl.col("start") + 1,
            _tss=pl.when(pl.col("strand") == "+")
                  .then(pl.col("start"))
                  .otherwise(pl.col("end")),
            _gene_strand=pl.col("strand"),
        )
        .select(["region", "gene_name", "gene_length", "_tss", "_gene_strand"])
        .unique(subset=["region"])
    )
 
 
def _gtf_mane_tss(gtf: pl.DataFrame) -> pl.DataFrame:
    """Per-gene TSS of the MANE Select transcript (GTF 'transcript' rows).
 
    For genes with no MANE Select transcript, falls back to the transcript
    tagged Ensembl_canonical. Used for dist_to_tss_v39 against a GENCODE v39
    GTF so the promoter region matches PromoterAI's benchmark definition.
 
    Returns one row per gene: region, _tss_v39, _gene_strand_v39.
    """
    tx = (
        gtf.filter(pl.col("feature") == "transcript")
        .with_columns(
            region=pl.col("attributes").str.extract(r'gene_id "([^"]+)"')
                  .str.split(".").list.first(),
            gene_type=pl.col("attributes")
                  .str.extract(r'gene_(?:type|biotype) "([^"]+)"'),
            _is_mane=pl.col("attributes").str.contains(r'tag "MANE_Select"'),
            _is_canonical=pl.col("attributes")
                  .str.contains(r'tag "Ensembl_canonical"'),
            _tss_v39=pl.when(pl.col("strand") == "+")
                  .then(pl.col("start"))
                  .otherwise(pl.col("end")),
            _gene_strand_v39=pl.col("strand"),
        )
        # Same protein-coding restriction as _gtf_gene_features: Ensembl_canonical
        # (the fallback) also tags lncRNAs / pseudogenes, which are not in scope.
        .filter(
            (pl.col("gene_type") == "protein_coding")
            & (pl.col("_is_mane") | pl.col("_is_canonical"))
        )
        # MANE Select wins over Ensembl_canonical; one transcript per gene.
        .sort(["region", "_is_mane"], descending=[False, True])
        .unique(subset=["region"], keep="first", maintain_order=True)
    )
    n_mane = int(tx.select(pl.col("_is_mane").sum()).item())
    logger.info(
        f"  v39 MANE/canonical TSS: {tx.height} genes "
        f"({n_mane} MANE Select, {tx.height - n_mane} Ensembl_canonical fallback)"
    )
    return tx.select(["region", "_tss_v39", "_gene_strand_v39"])
 
 
def _idempotent_join(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    on: list[str],
    how: str = "left",
    **kwargs,
) -> pl.LazyFrame:
    """Left-join that is safe to re-run in a notebook.
 
    Drops any columns the right frame would add that already exist in the left
    frame (excluding the join keys), so repeated cell executions don't produce
    DuplicateError.
    """
    left_cols = set(left.collect_schema().names())
    right_cols = set(right.collect_schema().names())
    keys = set(on) if isinstance(on, list) else {on}
    to_drop = (right_cols - keys) & left_cols
    if to_drop:
        left = left.drop(list(to_drop))
    return left.join(right, on=on, how=how, **kwargs)
 
 
def _next_inframe_atg_distance(seq: str, search_init: int = 3) -> int:
    """Return nt distance from position `search_init` to the next in-frame ATG.
 
    Searches codons starting at `search_init` (0-based, must be frame-0
    relative to the CDS start). Returns the codon offset (in nt) of the first
    ATG found, or -1 if none exists in `seq`.
    """
    for i in range(search_init, len(seq) - 2, 3):
        if seq[i:i + 3].upper() == "ATG":
            return i
    return -1
 
 
def _compute_next_in_frame(
    annos: pl.LazyFrame,
    tx_cds_len: pl.DataFrame,
    fasta_path: str,
) -> pl.LazyFrame:
    """Add next_in_frame_relative for start_lost SNVs (polars-native + pyfaidx).
 
    Port of add_more_annotations.py::_compute_next_in_frame. Identical sequence
    logic (genomic window from the CDS start, cds_length + 100 nt, reverse-
    complemented on the minus strand; next in-frame ATG / cds_length, capped
    at 1; 1.0 when no ATG, no CDS length or a FASTA error). Differences, all
    because the VEP output has one row per transcript:
      * cds_length is the transcript's CDS length (not a gene-level sum)
      * results are keyed and joined on (id, gene, feature), not (id, region)
      * cds_start is parsed from cds_position (see _cds_start_expr)
    """
    try:
        import pyfaidx  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "  pyfaidx not installed; skipping next_in_frame_relative. "
            "Add pyfaidx to your conda environment."
        )
        return annos
 
    schema = set(annos.collect_schema().names())
    req = {"id", "gene", "feature", "chrom", "pos", "ref", "alt",
           "cds_position", "strand", "consequence_start_lost"}
    missing = req - schema
    if missing:
        logger.warning(f"  next_in_frame: missing columns {missing}; skipping")
        return annos
 
    # ── Subset to start_lost SNVs with usable cds_start ──────────────────
    start_lost = (
        annos
        .filter(pl.col("consequence_start_lost") == 1)
        .filter(
            (pl.col("ref").str.len_chars() == 1) &
            (pl.col("alt").str.len_chars() == 1)
        )
        .with_columns(cds_start=_cds_start_expr())
        .filter(pl.col("cds_start").is_not_null())
        .filter(pl.col("strand").is_not_null())
        .select(["id", "gene", "feature", "chrom", "pos", "cds_start", "strand"])
        .collect()
    )
 
    if start_lost.is_empty():
        logger.info("  next_in_frame: no start_lost SNVs found; skipping")
        return annos
 
    # ── Per-transcript CDS lengths (from the GTF) ─────────────────────────
    start_lost = (
        start_lost.with_columns(_tx=_tx_key())
        .join(tx_cds_len, on="_tx", how="left")
        .drop("_tx")
    )
 
    # ── Fetch sequences and find next ATG ────────────────────────────────
    fasta = pyfaidx.Fasta(fasta_path, sequence_always_upper=True)
    records = []
    for row in start_lost.iter_rows(named=True):
        keys      = {"id": row["id"], "gene": row["gene"], "feature": row["feature"]}
        chrom     = str(row["chrom"])
        pos       = row["pos"]           # 1-based genomic position
        cds_pos   = row["cds_start"]     # 1-based position of variant IN CDS
        strand    = str(row["strand"])   # "1" or "-1"
        cds_len   = row.get("cds_length") or 0
 
        if cds_len == 0:
            records.append({**keys, "next_in_frame_relative": 1.0})
            continue
 
        try:
            chrom_key = chrom if chrom in fasta else chrom.lstrip("chr")
            if strand == "1":
                # Genomic start of CDS = variant_genomic_pos - (cds_pos - 1)
                cds_genome_start = pos - (cds_pos - 1)   # 1-based
                # Fetch CDS + 100 nt buffer for downstream search
                seq = str(fasta[chrom_key][cds_genome_start - 1 : cds_genome_start - 1 + cds_len + 100])
            else:
                # On minus strand VEP cds_pos counts from the transcript 5' end.
                # Genomic end of CDS (highest coordinate) = pos + (cds_pos - 1)
                cds_genome_end = pos + (cds_pos - 1)     # 1-based inclusive
                raw = str(fasta[chrom_key][cds_genome_end - cds_len - 100 : cds_genome_end])
                seq = raw[::-1].translate(str.maketrans("ACGTacgt", "TGCAtgca"))
 
            dist = _next_inframe_atg_distance(seq, search_init=3)
            rel  = 1.0 if dist < 0 else min(dist / cds_len, 1.0)
        except Exception as fe:
            logger.debug(f"  next_in_frame: FASTA error for {row['id']}: {fe}")
            rel = 1.0
 
        records.append({**keys, "next_in_frame_relative": rel})
 
    if not records:
        return annos
 
    nif = pl.DataFrame(records, schema={
        "id": pl.Utf8, "gene": pl.Utf8, "feature": pl.Utf8,
        "next_in_frame_relative": pl.Float32,
    })
    annos = _idempotent_join(
        annos, nif.lazy(), on=["id", "gene", "feature"], validate="1:1"
    )
    logger.info(f"  next_in_frame_relative OK ({len(records)} start_lost variant-transcripts)")
    return annos

# new 
def _tx_key(col: str = "feature") -> pl.Expr:
    """Transcript ID without version (ENST00000123456.7 -> ENST00000123456).
 
    Same version stripping add_more_annotations.py applies to gene_id.
    """
    return pl.col(col).str.split(".").list.first()
 
 
def _cds_start_expr() -> pl.Expr:
    """cds_start from VEP's cds_position string ("123", "123-125", "?-125").
 
    add_more_annotations.py reads VEP's integer `cds_start` field. Here VEP ran
    without --total_length, so cds_position holds just the position(s); the
    leading integer is cds_start ("?-125" -> null, like a missing cds_start).
    """
    return pl.col("cds_position").str.extract(r"^(\d+)").cast(pl.Int64)
 

def _gtf_transcript_cds(gtf: pl.DataFrame) -> pl.DataFrame:
    """Per-transcript CDS segments from GTF 'CDS' rows.

    Returns one row per CDS segment: _tx, chrom, start, end, strand, _cds_start_nf.
    Unlike add_more_annotations.py (gene-level, which double-counts exons shared
    by isoforms), this is keyed by transcript, matching VEP's `feature` column.
    chrY PAR copies are dropped so each transcript maps to one chromosome.
    """
    cds = (
        gtf.filter(pl.col("feature") == "CDS")
        .with_columns(
            transcript_id=pl.col("attributes").str.extract(r'transcript_id "([^"]+)"'),
            _cds_start_nf=pl.col("attributes").str.contains(r'tag "cds_start_NF"'),
        )
        .filter(
            pl.col("transcript_id").is_not_null()
            & ~pl.col("transcript_id").str.ends_with("_PAR_Y")
        )
        .with_columns(_tx=_tx_key("transcript_id"))
    )
    tx_chrom = (
        cds.select("_tx", "chrom").unique()
        .sort(["_tx", pl.col("chrom") == "chrY"])
        .unique(subset="_tx", keep="first", maintain_order=True)
    )
    return (
        cds.join(tx_chrom, on=["_tx", "chrom"], how="semi")
        .select("_tx", "chrom", "start", "end", "strand", "_cds_start_nf")
    )



 
def _gtf_transcript_cds_length(gtf: pl.DataFrame) -> pl.DataFrame:
    """Sum CDS exon lengths per transcript.
 
    Same as add_more_annotations.py's gene_cds_len (CDS rows, end - start + 1,
    summed), but grouped by transcript_id instead of gene_id: VEP output has
    one row per transcript, and a gene-level sum counts exons shared between
    isoforms once per isoform.
 
    Returns: _tx (ENST without version), cds_length.
    """
    per_chrom = (
        gtf.filter(pl.col("feature") == "CDS")
        .with_columns(
            transcript_id=pl.col("attributes").str.extract(r'transcript_id "([^"]+)"'),
            seg_len=pl.col("end") - pl.col("start") + 1,
        )
        .filter(
            pl.col("transcript_id").is_not_null()
            & ~pl.col("transcript_id").str.ends_with("_PAR_Y")
        )
        .with_columns(_tx=_tx_key("transcript_id"))
        .group_by(["_tx", "chrom"])
        .agg(cds_length=pl.col("seg_len").sum())
    )
    # Guard for PAR genes annotated on both chrX and chrY under one ID:
    # keep one copy (non-chrY) instead of summing both.
    return (
        per_chrom.sort(["_tx", pl.col("chrom") == "chrY"])
        .unique(subset=["_tx"], keep="first", maintain_order=True)
        .select(["_tx", "cds_length"])
    )

# EVA
def get_regions_positive_strand(variant_df, fasta_path):
    seqs_df = variant_df.query("strand == '1'").copy()
    if seqs_df.empty:
        for col, dtype in [("Start", np.int64), ("End", np.int64), ("index", np.int64)]:
            seqs_df[col] = pd.Series(dtype=dtype)
        seqs_df["Strand"] = pd.Series(dtype=object)
        seqs_df["seq"] = pd.Series(dtype=object)
        return seqs_df
    seqs_df["Start"] = seqs_df["pos"] - seqs_df["variant_pos_cds"]
    seqs_df["End"] = seqs_df["pos"] + 100
    seqs_df["Strand"] = seqs_df["strand"].replace({1: "+", -1: "-"})
    seqs_df["index"] = np.arange(len(seqs_df))
    seqs_pr = pr.PyRanges(seqs_df)
    seqs_pr.seq = pr.get_sequence(seqs_pr, path=fasta_path)
    return seqs_pr.df.copy()


def get_regions_negative_strand(variant_df, fasta_path):
    seqs_df = variant_df.query("strand == '-1'").copy()
    if seqs_df.empty:
        for col, dtype in [("End", np.int64), ("Start", np.int64), ("index", np.int64)]:
            seqs_df[col] = pd.Series(dtype=dtype)
        seqs_df["Strand"] = pd.Series(dtype=object)
        seqs_df["seq"] = pd.Series(dtype=object)
        return seqs_df
    seqs_df["End"] = seqs_df["pos"] + seqs_df["variant_pos_cds"] - 1
    seqs_df["Start"] = seqs_df["End"] - seqs_df["cds_length"]
    seqs_df["Strand"] = seqs_df["strand"].replace({1: "+", -1: "-"})
    seqs_df["index"] = np.arange(len(seqs_df))
    seqs_pr = pr.PyRanges(seqs_df)
    seqs_pr.seq = pr.get_sequence(seqs_pr, path=fasta_path)
    return seqs_pr.df.copy()


def next_inframe_start_codon_distance(seq, search_init=0):
    """Distance to the next in-frame ATG, or -1 if none found."""
    pos = seq.find("ATG", search_init)
    if pos == -1:
        return -1
    rel = pos - search_init
    if rel % 3 == 0:
        return pos
    next_init = pos + (3 - rel % 3)
    if next_init >= len(seq):
        return -1
    return next_inframe_start_codon_distance(seq, next_init)


def convert_to_int_and_get_max(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        if value is None:
            return None
        parts = str(value).split("-")
        ints = [int(v) for v in parts if v.isdigit()]
        return max(ints) if ints else None

# ------------- 1. concatenate all shards of annotations into a single DataFrame and write to a parquet file
def concat_annotations(shards_dir: list[str], out_file: str):
    """Concatenate shards of annotations into a single parquet file."""
    all_shards = []
    for f in shards_dir:
        all_shards.extend(
            glob.glob(os.path.join(f, "shard-*", "subshard-*", "annotations.parquet"))
        )

    if not all_shards:
        raise FileNotFoundError(f"No annotation parquet files found in: {shards_dir}")

    logger.info(f"Found {len(all_shards)} parquet files")
    logger.info(f"Writing concatenated annotations to {out_file}")
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_parquet(all_shards).drop(ANN_COLUMNS).drop(VEP_TO_DROP)
    names = lf.collect_schema().names()

    casts = {
        **{c: t for c, t in BASE_CASTS.items() if c in names},
        **{c: pl.Float32 for c in names if c.startswith("gnomadg")},
    }

    (
        lf.unique(subset=["ID", "feature"])
        .filter(
            (
                pl.col("ALT").str.len_chars().cast(pl.Int64)
                - pl.col("REF").str.len_chars().cast(pl.Int64)
            ).abs() < 50
        )
        .with_columns(
            [pl.col(c).cast(t, strict=False) for c, t in casts.items()]
        )
        .sink_parquet(out_file, engine="streaming")
    )

    logger.info("Finished concatenating annotations")

# ------------- 2. write variant metadata to a parquet file
def write_variant_metadata(annotations_file: str, out_file: str):
    annos = pl.scan_parquet(annotations_file)
    annos = annos.rename({c: c.lower() for c in annos.collect_schema().names()})

    vm = annos.select(["id", "chrom", "pos", "ref", "alt"]).unique(subset="id")

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    vm.sink_parquet(out_file, engine="streaming")

# ------------- 3. process VEP annotations
def process_vep(
    annotation_file,
    variant_metadata_file,
    fasta_path,
    gtf_path,
    blosum_path,
    output_path,
    gene_filters=None,
    n_samples=None,
    sanity_check=False,
):
    """
    # annotations contains CHROM POS REF ALT ID CONSEQUENCE etc
    Processes VEP annotations and computes all annotations that only require
    VEP output + variant metadata (no external score files):
      - Consequence dummy variables (one-hot)
      - LOFTEE HC / LC flags
      - rename AF to MAF
      - Relative CDS position
      - Start-lost next in-frame ATG distance
      - SpliceAI max delta score
      - NOT DONE 5' UTR variant consequence dummies
      - Variant length / is_indel / is_insertion / is_deletion flags
      - Distance to TSS (from GTF)

    Writes output to `output_path`.
    """
    logger.info(f"Loading VEP annotations from {annotation_file}")
    vep_file = pl.scan_parquet(annotation_file)

    logger.info("Standardizing column names to lowercase")
    vep_file = vep_file.rename(
        {col: col.lower() for col in vep_file.collect_schema().names()}
    )

    # ── Gene / biotype filters ─────────────────────────────────────────────
    if gene_filters is not None:
        genes_to_keep_file = gene_filters.get("genes_to_keep_file", None)
        if genes_to_keep_file is not None:
            logger.info(f"Filtering for genes in file: {genes_to_keep_file}")
            gdf = pl.read_parquet(genes_to_keep_file)
            gene_col = next(
                (c for c in ["gene_id", "gene", "region"] if c in gdf.columns), None
            )
            if gene_col is None:
                raise ValueError(f"No valid gene column in {genes_to_keep_file}")
            genes_to_keep = gdf[gene_col].unique()
            vep_file = vep_file.filter(pl.col("gene").is_in(genes_to_keep))

        biotypes = gene_filters.get("biotypes", None)
        if biotypes is not None:
            logger.info(f"Filtering for biotypes: {biotypes}")
            vep_file = vep_file.filter(pl.col("biotype").is_in(biotypes))

    if sanity_check:
        for col in ("chrom", "pos", "ref", "alt"):
            assert vep_file.select(pl.col(col).is_null().sum()).collect().item() == 0
    
    # ── Consequence dummies ────────────────────────────────────────────────
    logger.info("Creating consequence dummy variables")
    vep_file = vep_file.with_row_index("row_nr")
    dummies = (
        vep_file.select(["row_nr", "consequence"])
        .with_columns(pl.col("consequence").str.split("&"))
        .explode("consequence")
        .filter(pl.col("consequence").is_not_null() & (pl.col("consequence") != ""))
        .collect()
        .to_dummies(columns="consequence")
        .group_by("row_nr")
        .max()
    )
    logger.info(f"Created {len(dummies.columns) - 1} consequence dummy columns")
    vep_file = vep_file.join(dummies.lazy(), on="row_nr", how="left").drop("row_nr")
    vep_file = vep_file.rename(
        {col: col.lower() for col in vep_file.collect_schema().names()}
    )

    dtype_update = {
        col: pl.Int8
        for col in vep_file.collect_schema().names()
        if "consequence_" in col
    }
    dtype_update["strand"] = pl.Utf8
    vep_file = vep_file.with_columns(
        [pl.col(col).cast(dtype) for col, dtype in dtype_update.items()]
    )

    # Canonical variant ID
    vep_file = vep_file.with_columns(
        pl.concat_str(
            [pl.col("chrom"), pl.col("pos"), pl.col("ref"), pl.col("alt")],
            separator=":",
        ).alias("id")
    )

    logger.info("Removing duplicate entries")
    # vep_file = vep_file.unique()
    vep_file = vep_file.unique(
        subset=["id", "gene", "feature"]
    )

    # ── Load variant metadata ──────────────────────────────────────────────
    logger.info("Loading variant metadata")
    variant_metadata = pl.scan_parquet(variant_metadata_file)
    annos = vep_file

    # ── LOFTEE ────────────────────────────────────────────────────────────
    logger.info("Processing LOFTEE annotations")
    annos = annos.with_columns(
        pl.when(pl.col("lof") == "HC")
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("loftee_hc"),
        pl.when(pl.col("lof").is_null())
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("loftee_hc_is_na"),
        pl.when(pl.col("lof") == "LC")
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("loftee_lc"),
        pl.when(pl.col("lof").is_null())
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("loftee_lc_is_na"),
    )

    # ── MAF ───────────────────────────────────────────────────────────────
    logger.info("renaming MAF")
    annos = annos.rename({"af": "maf_cohort"})
    # vm_schema = variant_metadata.collect_schema().names()
    # if "mac_cohort" in vm_schema and n_samples is not None:
    #     logger.info(f"{n_samples} samples, computing MAF from MAC")
    #     cols_to_select = [c for c in vm_schema if "cohort" in c]
    #     maf_df = variant_metadata.select(["id"] + cols_to_select).with_columns(
    #         maf_cohort=pl.col("mac_cohort") / (2 * n_samples)
    #     )
    #     annos = annos.join(maf_df, on="id", how="left", validate="m:1")
    # -- MAF max of all gnomad populations  --------------------
    logger.info("Computing MAF from gnomAD populations")
    gnomad_cols = [
        c for c in annos.collect_schema().names()
        if c.startswith(("gnomadg_af_"))
    ]
    annos = annos.with_columns(
        maf_gnomad=pl.max_horizontal(
            [pl.col(c).cast(pl.Float32, strict=False) for c in gnomad_cols]
        )
    )

    tmp = Path(output_path).with_suffix(".stage1.parquet")
    annos.sink_parquet(tmp, engine="streaming")
    annos = pl.scan_parquet(tmp).drop(gnomad_cols)

    # -------- was not run with --total_length, so cds_position is just the start (or range) of the variant in the CDS
        # ── GENCODE v49 GTF (VEP 115) ─────────────────────────────────────────
    logger.info(f"Loading GTF {gtf_path}")
    gtf = _load_gtf_polars(gtf_path)
    genes = _gtf_gene_features(gtf)
    tx_cds_len = _gtf_transcript_cds_length(gtf)
    del gtf
    gc.collect()
    logger.info(
        f"  GTF: {genes.height} protein-coding genes, "
        f"{tx_cds_len.height} transcripts with CDS"
    )

    # ── relative_cds_position: cds_start / total CDS length ───────────────
    # As in add_more_annotations.py, but the CDS length is per transcript
    # (joined on `feature`), because there is one row per transcript.
    logger.info("Computing relative CDS positions (cds_start / GTF CDS length)")
    annos = (
        _idempotent_join(
            annos.with_columns(_tx=_tx_key()), tx_cds_len.lazy(),
            on=["_tx"], validate="m:1",
        )
        .with_columns(
            relative_cds_position=(
                _cds_start_expr().cast(pl.Float64)
                / pl.col("cds_length").cast(pl.Float64)
            ).clip(0.0, 1.0).round(4).cast(pl.Float32)
        )
        .drop("_tx")
        # keep cds_length — also used by next_in_frame
    )
    match_rate = (
        annos.filter(pl.col("cds_position").is_not_null())
        .select(pl.col("cds_length").is_not_null().mean())
        .collect()
        .item()
    )
    logger.info(f"  CDS length found for {match_rate:.1%} of rows with cds_position")

    # ── next_in_frame_relative (start_lost SNVs only) ────────────────────
    logger.info("Processing start_lost variants for next in-frame ATG")
    annos = _compute_next_in_frame(annos, tx_cds_len, fasta_path)


    # ── Relative CDS position ─────────────────────────────────────────────
    # does not work this way, as not run with --total_length
    # logger.info("Computing relative CDS positions")
    # sites = (
    #     annos.with_columns(cds_parts=pl.col("cds_position").str.split("/"))
    #     .with_columns(
    #         length=pl.col("cds_parts").list.get(1),
    #         protein_pos=pl.col("cds_parts").list.get(0),
    #     )
    #     .with_columns(
    #         pl.col("protein_pos")
    #         .map_elements(convert_to_int_and_get_max, return_dtype=pl.Int64)
    #         .alias("protein_pos")
    #     )
    #     .filter(pl.col("protein_pos").is_not_null())
    #     .with_columns(pl.col("length").cast(pl.Int64))
    #     .with_columns(
    #         (pl.col("protein_pos") / pl.col("length"))
    #         .round(2)
    #         .alias("relative_cds_position")
    #     )
    # )
    # cds_merged = annos.select("id", "gene", "feature").join(
    #     sites.select("id", "gene", "feature", "relative_cds_position"),
    #     on=["id", "gene", "feature"],
    #     how="left",
    # )
    # annos = annos.join(cds_merged, on=["id", "gene", "feature"], how="left", validate="1:1")

    # ── Start-lost: next in-frame ATG ─────────────────────────────────────
    # logger.info("Processing start_lost variants for next in-frame ATG")
    # schema_names = annos.collect_schema().names()
    # if "consequence_start_lost" in schema_names:
    #     vep_start_lost = (
    #         annos.filter(pl.col("consequence_start_lost") == 1)
    #         .select(
    #             [
    #                 "pos",
    #                 "chrom",
    #                 "gene",
    #                 "id",
    #                 "cds_position",
    #                 "codons",
    #                 "strand",
    #                 "allele",
    #             ]
    #         )
    #         .collect()
    #         .to_pandas()
    #     )
    #     vep_start_lost_snv = vep_start_lost[
    #         vep_start_lost["allele"].str.len() == 1
    #     ].copy()
    #     vep_start_lost_snv[["variant_pos_cds", "cds_length"]] = vep_start_lost_snv[
    #         "cds_position"
    #     ].str.split("/", expand=True)
    #     vep_start_lost_snv = vep_start_lost_snv[
    #         vep_start_lost_snv["variant_pos_cds"].str.len() == 1
    #     ]
    #     vep_start_lost_snv["variant_pos_cds"] = vep_start_lost_snv[
    #         "variant_pos_cds"
    #     ].astype(int)
    #     vep_start_lost_snv["cds_length"] = vep_start_lost_snv["cds_length"].astype(int)
    #     vep_start_lost_snv["Chromosome"] = vep_start_lost_snv["chrom"].str.replace(
    #         r"^(?!chr)", "chr", regex=True
    #     )

    #     logger.info(f"Processing {len(vep_start_lost_snv)} start_lost SNVs")
    #     strands_pos = get_regions_positive_strand(vep_start_lost_snv, fasta_path)
    #     strands_neg = get_regions_negative_strand(vep_start_lost_snv, fasta_path)
    #     start_lost_df = pd.concat([strands_pos, strands_neg])
    #     start_lost_df["next_in_frame"] = start_lost_df.apply(
    #         lambda x: next_inframe_start_codon_distance(x["seq"], search_init=3), axis=1
    #     )
    #     start_lost_df["next_in_frame_relative"] = (
    #         start_lost_df["next_in_frame"] / start_lost_df["cds_length"]
    #     )
    #     start_lost_df.loc[
    #         start_lost_df["next_in_frame_relative"] < 0, "next_in_frame_relative"
    #     ] = 1
    #     start_lost_pl = pl.DataFrame(
    #         start_lost_df[["gene", "id", "next_in_frame_relative"]]
    #     )
    #     annos = annos.join(
    #         start_lost_pl.lazy(), on=["id", "gene", "feature"], how="left", validate="1:1"
    #     )

    # # ── SpliceAI max delta score ───────────────────────────────────────────
    # logger.info("Processing SpliceAI predictions")
    # if "spliceai_pred" in annos.collect_schema().names():
    #     annos = annos.with_columns(
    #         pl.col("spliceai_pred")# there are multiple predictions
    #         .str.split("|")
    #         .list.slice(1, 4)
    #         .list.eval(pl.element().cast(pl.Float32, strict=False))
    #         .list.max()
    #         .alias("spliceai_delta_score")
    #     ).drop("spliceai_pred")

     # ── SpliceAI max delta score ───────────────────────────────────────────
    # SpliceAI comes as separate columns: spliceai_pred_ds_{ag,al,dg,dl} (delta scores, 0-1) and spliceai_pred_dp_{ag,al,dg,dl} (positions). 
    logger.info("Processing SpliceAI predictions")
    schema_names = annos.collect_schema().names()
    ds_cols = [f"spliceai_pred_ds_{s}" for s in SPLICE if f"spliceai_pred_ds_{s}" in schema_names]
    if ds_cols:
        annos = annos.with_columns(
            spliceai_delta_score=pl.max_horizontal(
                [pl.col(c).cast(pl.Float32, strict=False) for c in ds_cols]
            )
        ).drop([c for c in schema_names if c.startswith("spliceai_pred_")])
        logger.info(f"  spliceai_delta_score from {ds_cols}")
    else:
        logger.warning("  No spliceai_pred_ds_* columns found; skipping SpliceAI")
 

    # ── BLOSUM62 scores for missense substitutions ────────────────────────
    logger.info("Computing BLOSUM62 scores")
    if "amino_acids" in annos.collect_schema().names():
        _BLOSUM62_DF = _load_blosum62_sij(blosum_path)
        annos = (
            annos.with_columns(
                pl.col("amino_acids")
                .str.extract(r"^([A-Z])\/([A-Z])$", group_index=1)
                .alias("_ref_aa"),
                pl.col("amino_acids")
                .str.extract(r"^([A-Z])\/([A-Z])$", group_index=2)
                .alias("_alt_aa"),
            )
            .join(
                _BLOSUM62_DF.lazy(),
                left_on=["_ref_aa", "_alt_aa"],
                right_on=["ref_aa", "alt_aa"],
                how="left",
            )
            .drop(["_ref_aa", "_alt_aa"])
        )

    # ── 5' UTR dummies ────────────────────────────────────────────────────
    # have only: 5utr_annotation, 5utr_consequence
    # logger.info("Processing 5' UTR variant consequence annotations")
    # if "five_prime_utr_variant_consequence" in annos.collect_schema().names():
    #     utr_df = annos.select(
    #         ""id", "gene", "feature" "five_prime_utr_variant_consequence"
    #     ).with_row_index("row_nr")
    #     dummies = (
    #         utr_df.select(["row_nr", "five_prime_utr_variant_consequence"])
    #         .with_columns(pl.col("five_prime_utr_variant_consequence").str.split("&"))
    #         .explode("five_prime_utr_variant_consequence")
    #         .filter(
    #             pl.col("five_prime_utr_variant_consequence").is_not_null()
    #             & (pl.col("five_prime_utr_variant_consequence") != "")
    #         )
    #         .collect()
    #         .to_dummies(columns="five_prime_utr_variant_consequence")
    #         .group_by("row_nr")
    #         .max()
    #     )
    #     utr_df = (
    #         utr_df.select("id", "gene", "feature", "row_nr")
    #         .join(dummies.lazy(), on="row_nr", how="left")
    #         .drop("row_nr")
    #     )
    #     annos = annos.join(utr_df, on=["id", "gene", "feature"], how="left", validate="1:1")

    # ── Variant length / indel flags ──────────────────────────────────────
    logger.info("Computing variant lengths and indel flags")
    annos = annos.with_columns(
        pl.max_horizontal(
            [pl.col("ref").str.len_chars(), pl.col("alt").str.len_chars()]
        ).alias("variant_length")
    ).with_columns(
        is_indel=(pl.col("variant_length") > 1).cast(pl.Int8),
        is_insertion=(
            pl.col("ref").str.len_chars() < pl.col("alt").str.len_chars()
        ).cast(pl.Int8),
        is_deletion=(
            pl.col("ref").str.len_chars() > pl.col("alt").str.len_chars()
        ).cast(pl.Int8),
    )

    # TODO: check if this is needed
    # # ── Set region = gene (required for TSS and downstream joins) ─────────
    # annos = annos.with_columns(region=pl.col("gene"))

    # # ── dist_to_tss, gene_length, gene_name (add_more_annotations.py step0b) ──
    # # One TSS per gene; each transcript row gets its gene's TSS (m:1 join).
    # logger.info("Calculating distance to TSS")
    # annos = (
    #     _idempotent_join(annos, genes.lazy(), on=["region"], validate="m:1")
    #     .with_columns(
    #         dist_to_tss=pl.when(pl.col("_gene_strand") == "+")
    #         .then(pl.col("pos") - pl.col("_tss"))
    #         .otherwise(pl.col("_tss") - pl.col("pos"))
    #     )
    #     .drop(["_tss", "_gene_strand"])
    # )
    # logger.info("  dist_to_tss / gene_length / gene_name OK")

    # logger.info(f"Writing VEP-processed annotations to {output_path}")
    # Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # annos.sink_parquet(output_path, engine="streaming")
    # logger.info("VEP processing complete")
    # tmp.unlink(missing_ok=True)

    # ── Set region = gene (required for TSS and downstream joins) ─────────
    annos = annos.with_columns(region=pl.col("gene"))

    # ── Distance to TSS ───────────────────────────────────────────────────
    logger.info("Calculating distance to TSS")
    gencode_pr = pr.read_gtf(gtf_path, as_df=True)
    gencode_pl = pl.from_pandas(gencode_pr).filter(
        pl.col("gene_type") == "protein_coding"
    )
    gencode_genes = gencode_pl.filter(pl.col("Feature") == "gene").with_columns(
        region=pl.col("gene_id").str.split(".").list.first()
    )
    tss_df = gencode_genes.with_columns(
        gene_length=pl.col("End") - pl.col("Start") + 1,
        tss=pl.when(pl.col("Strand") == "+")
        .then(pl.col("Start"))
        .otherwise(pl.col("End")),
    ).select(["tss", "Strand", "gene_length", "region"]).unique(subset="region")

    logger.info("Joining distance to TSS")
    anno_tss = (
        annos.select(["id", "pos", "region"])
        .join(tss_df.lazy(), on="region", how="left")
        .with_columns(
            dist_to_tss=pl.when(pl.col("Strand") == "+")
            .then(pl.col("pos") - pl.col("tss"))
            .otherwise(pl.col("tss") - pl.col("pos"))
        )
        # .collect(engine="streaming")
    )
    anno_tss = anno_tss.rename({col: col.lower() for col in anno_tss.columns})
    annos = annos.join(anno_tss.lazy(), on=["id", "pos", "region"], how="left")
    annos = annos.unique(subset=["id", "gene", "region", "feature"])

    logger.info(f"Writing VEP-processed annotations to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    annos.sink_parquet(output_path, engine="streaming")
    logger.info("VEP processing complete")
    tmp.unlink(missing_ok=True)



# ── Step 2b: GPN-MSA ──────────────────────────────────────────────────────

def merge_gpn_msa(variant_metadata_path, scores_gpn_msa_file, output_path):
    """Merge GPN-MSA scores for all chromosomes (one job)."""
    logger.info("Loading annotations for GPN-MSA")
    vm = pl.read_parquet(variant_metadata_path, columns=["id", "chrom"]).with_columns(
        pl.col("chrom").cast(pl.String)
    )

    scores_lazy = pl.scan_parquet(scores_gpn_msa_file)
    unique_chroms = vm["chrom"].unique().to_list()
    logger.info(f"Processing GPN-MSA for {len(unique_chroms)} chromosomes")

    all_scores = []
    for chrom in unique_chroms:
        gpn_chrom = chrom.lstrip("chr")
        scores_chr = (
            scores_lazy.filter(pl.col("chrom").str.to_lowercase() == gpn_chrom.lower())
            .with_columns(pl.lit(f"chr{gpn_chrom}").alias("chrom"))
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("chrom"),
                        pl.col("pos").cast(pl.Utf8),
                        pl.col("ref"),
                        pl.col("alt"),
                    ],
                    separator=":",
                ).alias("id")
            )
            .select("id", "gpn_score")
        )
        sampled_chrom = vm.filter(pl.col("chrom").str.to_lowercase() == chrom.lower()).select("id").lazy()
        merged = sampled_chrom.join(scores_chr, on="id", how="left").collect()
        all_scores.append(merged)
        logger.info(f"  GPN-MSA done for {chrom}")

    all_scores_df = pl.concat(all_scores)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    all_scores_df.write_parquet(output_path)
    logger.info(f"GPN-MSA scores written to {output_path}")




# ── Step 3: Merge all pre-CADD ─────────────────────────────────────────────

def merge_all_pre_cadd(
    vep_processed_path,
    # absplice2_path,
    gpn_msa_path,
    # abexp_path,
    # protein_domains_path,
    # cpt1_path,
    # promoter_ai_file,
    # pai3_file,
    # encode_path,
    # output_path,
):
    """
    Merge all external scores (except CADD) into the main annotation table:
      AbSplice2, GPN-MSA, AbExp, PromoterAI, PrimateAI3D,
      protein domains (MobiDB/TED/LC), CPT-1, ENCODE.
    Writes to output_path (the pre-CADD intermediate).
    """
    logger.info("Loading VEP-processed annotations")
    # chrX PAR-region variants can appear twice in VEP output (only tss /
    # dist_to_tss may differ between copies).  Deduplicate eagerly so all
    # downstream 1:1 join validations pass.
    # sanity checked this using
    # annos = pl.read_parquet(vep_processed_path)
    # cnts = annos.group_by('id', 'gene').len().sort('len')
    # annos.join(cnts.filter(pl.col('len') > 1).select('id'), on='id', how='semi') \
    #     .sort("id") \
    #     .select(['id', 'gene', 'feature', 'chrom', 'pos', 'dist_to_tss'])\
    #     .select('chrom').unique()
    # # only returns X
    annos = (
        pl.read_parquet(vep_processed_path)
        .sort("dist_to_tss", nulls_last=True)
        .unique(subset=["id", "feature"], keep="first")
        .lazy()
    )

    # ── AbSplice2 ─────────────────────────────────────────────────────────
    # logger.info("Merging AbSplice2 scores")
    # # AbSplice2 also has fully-identical duplicate rows for chrX PAR variants.
    # absplice2 = pl.scan_parquet(absplice2_path).unique()
    # annos = annos.join(
    #     absplice2,
    #     on=["chrom", "pos", "ref", "alt", "gene"],
    #     how="left",
    #     validate="1:1",
    # ).rename({col: col.lower() for col in absplice2.collect_schema().names()})

    # ── GPN-MSA ───────────────────────────────────────────────────────────
    logger.info("Merging GPN-MSA scores")
    gpn_msa = pl.scan_parquet(gpn_msa_path)
    annos = annos.join(gpn_msa, on="id", how="left", validate="m:1")

    # # ── AbExp ─────────────────────────────────────────────────────────────
    # logger.info("Merging AbExp scores")
    # abexp = pl.scan_parquet(abexp_path).drop(["chrom", "ref", "alt", "pos"])
    # annos = annos.join(abexp, on=["id", "region"], how="left", validate="1:1")

    # # ── PromoterAI ────────────────────────────────────────────────────────
    # logger.info("Merging PromoterAI scores")
    # promoter_ai = (
    #     pl.scan_parquet(promoter_ai_file)
    #     .with_columns(
    #         pl.concat_str(
    #             [
    #                 pl.col("chrom"),
    #                 pl.col("pos").cast(pl.Utf8),
    #                 pl.col("ref"),
    #                 pl.col("alt"),
    #             ],
    #             separator=":",
    #         ).alias("id")
    #     )
    #     .rename({"gene": "gene_name", "gene_id": "gene", "promoterAI": "promoterai"})
    #     .with_columns(pl.col("promoterai").abs().alias("promoterai_abs"))
    #     .select("promoterai", "promoterai_abs", "id", "gene", "feature")
    # )
    # scores = annos.select("gene", "id").join(
    #     promoter_ai, how="left", on=["id", "gene", "feature"], validate="1:m"
    # )
    # scores_grouped = scores.group_by(["gene", "id"]).agg(
    #     pl.col("promoterai").sort_by(pl.col("promoterai_abs"), descending=True).first()
    # )
    # annos = annos.join(
    #     scores_grouped.select("promoterai", "id", "gene", "feature"),
    #     how="left",
    #     on=["id", "gene", "feature"],
    #     validate="1:1",
    # )

    # # ── PrimateAI-3D ─────────────────────────────────────────────────────
    # logger.info("Merging PrimateAI-3D scores")
    # paidf = pl.scan_parquet(pai3_file)
    # annos = annos.join(paidf, how="left", on=["id", "feature"], validate="1:1")

    # # ── Protein domains ───────────────────────────────────────────────────
    # logger.info("Merging protein domain annotations")
    # all_domains = pl.read_parquet(protein_domains_path)
    # schema_names = annos.collect_schema().names()

    # if "protein_position" in schema_names:
    #     anno_coding = (
    #         annos.select(["id", "region", "protein_position"])
    #         .drop_nulls("protein_position")
    #         .with_columns(pos_raw=pl.col("protein_position").str.split("/").list.get(0))
    #         .with_columns(split_struct=pl.col("pos_raw").str.split_exact("-", 1))
    #         .with_columns(
    #             aa_start=pl.col("split_struct")
    #             .struct.field("field_0")
    #             .cast(pl.Int32, strict=False),
    #             aa_end=pl.col("split_struct")
    #             .struct.field("field_1")
    #             .fill_null(pl.col("split_struct").struct.field("field_0"))
    #             .cast(pl.Int32, strict=False),
    #         )
    #         .select(["id", "region", "aa_start", "aa_end"])
    #         .collect()
    #     )

    #     overlap_df = (
    #         anno_coding.lazy()
    #         .join(all_domains.lazy(), on="region", how="inner")
    #         .filter(
    #             (
    #                 (pl.col("aa_start") <= pl.col("domain_end"))
    #                 & (pl.col("aa_start") >= pl.col("domain_start"))
    #             )
    #             | (
    #                 (pl.col("aa_end") <= pl.col("domain_end"))
    #                 & (pl.col("aa_end") >= pl.col("domain_start"))
    #             )
    #         )
    #         .group_by(["id", "region", "aa_start", "aa_end"])
    #         .agg(
    #             pl.col("mobi_full_disorder_priority").max(),
    #             pl.col("mobi_curated_disorder_priority").max(),
    #             pl.col("mobi_full_lip_priority").max(),
    #             pl.col("ted_domain").max(),
    #             pl.col("low_complexity_domain").max(),
    #         )
    #         # Collapse to unique (id, region) pairs
    #         .group_by(["id", "region"])
    #         .agg(
    #             pl.col("mobi_full_disorder_priority").max(),
    #             pl.col("mobi_curated_disorder_priority").max(),
    #             pl.col("mobi_full_lip_priority").max(),
    #             pl.col("ted_domain").max(),
    #             pl.col("low_complexity_domain").max(),
    #         )
    #         .collect()
    #     )

    #     domain_annos = (
    #         annos.select(["id", "region"])
    #         .join(overlap_df.lazy(), on=["id", "region"], how="left")
    #         .with_columns(
    #             pl.col("mobi_full_disorder_priority").fill_null(False),
    #             pl.col("mobi_curated_disorder_priority").fill_null(False),
    #             pl.col("mobi_full_lip_priority").fill_null(False),
    #             pl.col("ted_domain").fill_null(False),
    #             pl.col("low_complexity_domain").fill_null(False),
    #         )
    #     )
    #     annos = annos.join(
    #         domain_annos, on=["id", "region"], how="left", validate="1:1"
    #     )

    # # ── CPT-1 ─────────────────────────────────────────────────────────────
    # logger.info("Merging CPT-1 scores")
    # cpt1_scores = pl.scan_parquet(cpt1_path)
    # if (
    #     "protein_position" in annos.collect_schema().names()
    #     and "amino_acids" in annos.collect_schema().names()
    # ):
    #     annos = (
    #         annos.with_columns(
    #             prot_pos=pl.col("protein_position").str.split("/").list.get(0)
    #         )
    #         .join(cpt1_scores, on=["region", "amino_acids", "prot_pos"], how="left")
    #         .drop("prot_pos")
    #     )

    # # ── ENCODE ────────────────────────────────────────────────────────────
    # # ENCODE is position-based (not gene-specific) → join on id only.
    # logger.info("Merging ENCODE cCRE annotations")
    # encode_bool_cols = list(CCRE_COLUMN_MAP.values())
    # encode_df = pl.scan_parquet(encode_path).select(["id"] + encode_bool_cols)
    # annos = annos.join(encode_df, on="id", how="left", validate="m:1")

    logger.info(f"Writing pre-CADD annotations to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    annos.sink_parquet(output_path, engine="streaming")
    logger.info("Pre-CADD merge complete")


# ── Step 4: CADD ───────────────────────────────────────────────────────────


def merge_cadd(nocadd_path, cadd_file, annotation_columns, output_path):
    """
    Merge CADD scores and write the final annotation file.
    Casts all float64 columns to float32 to reduce disk size.
    """
    logger.info("Loading pre-CADD annotations")
    gc.collect()
    annos = pl.scan_parquet(nocadd_path)

    logger.info(f"Loading CADD scores from {cadd_file}")
    cadd = (
        pl.scan_parquet(cadd_file)
        .with_columns(
            pl.concat_str(
                [
                    "chr" + pl.col("#Chrom"),
                    pl.col("Pos"),
                    pl.col("Ref"),
                    pl.col("Alt"),
                ],
                separator=":",
            ).alias("id"),
            pl.col("GeneID").alias("region"),
        )
        .drop(["#Chrom", "Pos", "Ref", "Alt", "GeneID"])
    )
    to_lower_renamer = {i: i.lower() for i in cadd.columns}
    cadd = cadd.rename(to_lower_renamer)
    col_set = list(set(["id", "region"] + annotation_columns))
    col_set = [col for col in col_set if col in cadd.columns]
    logger.info(f"Joining columns from cadd: {col_set}")
    gene_set = set(annos.select(pl.col("region").unique()).collect().to_series())
    logger.info(f"Filtering CADD for {len(gene_set)} genes")
    cadd_small = cadd.select(*col_set).filter(pl.col("region").is_in(gene_set))

    logger.info("Joining CADD scores")
    annos = annos.join(cadd_small, on=["id", "region"], how="left")
    annos = annos.rename({col: col.lower() for col in annos.collect_schema().names()})

    logger.info("Casting float64 columns to float32")
    float64_cols = [name for name, dtype in annos.schema.items() if dtype == pl.Float64]
    annos = annos.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols])

    logger.info(f"Writing final annotations to {output_path}")
    annos.sink_parquet(output_path, engine="streaming")
    logger.info("Done")


# ── Step 5: Fill nulls ──────────────────────────────────────────────────────


def fill_nulls(input_path, annotation_specs, cols_to_keep, output_path):
    """
    Fill missing values, add per-column _is_na indicators, and select the
    final output schema.

    annotation_specs: dict mapping column name → [fill_value]  (from config)
    cols_to_keep: list of columns to keep in the output
    Consequence columns (consequence_*) are always filled with 0; no _is_na
    indicator is added for them.
    """
    fill_value_mapping = {col: vals[0] for col, vals in annotation_specs.items()}

    logger.info(f"Reading annotations from {input_path}")
    annos = pl.scan_parquet(input_path)

    logger.info("Filling consequence columns with 0")
    consequence_cols = [
        col for col in annos.collect_schema().names() if col.startswith("consequence_")
    ]
    consequence_col_fills = {col: 0 for col in consequence_cols}
    fill_vals = {**fill_value_mapping, **consequence_col_fills}

    all_cols = cols_to_keep + consequence_cols
    schema_cols = set(annos.collect_schema().names())
    missing = set(all_cols) - schema_cols
    if missing:
        logger.warning(
            f"Columns in config but absent from parquet (will be skipped): {missing}"
        )
    all_cols = [c for c in all_cols if c in schema_cols]
    logger.info(f"Selecting {len(all_cols)} cols")
    annos = annos.select(all_cols)

    # Only fill columns that are actually present after selection (guards against
    # annotation_specs entries not listed in annotation_columns).
    schema_names = set(annos.collect_schema().names())
    fill_vals = {col: val for col, val in fill_vals.items() if col in schema_names}
    logger.info(f"Filling {len(fill_vals)} columns")

    logger.info("Filling nulls and adding _is_na indicators")
    for col, fill_val in fill_vals.items():
        if "consequence_" not in col:
            annos = annos.with_columns(
                pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_is_na")
            )
        annos = annos.with_columns(pl.col(col).fill_null(fill_val).alias(col))

    logger.info(f"Writing fill-null annotations to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    annos.sink_parquet(output_path, engine="streaming")
    logger.info("Done")


# main function to run all steps
def main(config_path, config_general_path):
    # Paths
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with open(config_general_path, "r") as f:
        config_general = yaml.safe_load(f)

    # Input paths
    input_files = config["input_files"]

    SHARDS_DIR = [
        os.path.expanduser(path)
        for path in input_files["SHARDS_DIR"]
    ]

    BLOSUM_PATH = input_files["blosum_path"]
    FASTA_PATH = input_files["FASTA_PATH"]
    GTF_PATH = input_files["GTF_PATH"]
    CADD_PATH = input_files["CADD_PATH"]

    # Output paths
    output_files = config["output_files"]

    OUT_CONCAT_ANNOTATIONS = os.path.expanduser(
        output_files["OUT_CONCAT_ANNOTATIONS"]
    )
    OUT_VAR_METADATA = os.path.expanduser(
        output_files["OUT_VAR_METADATA"]
    )
    OUT_PROCESS_VEP = os.path.expanduser(
        output_files["OUT_PROCESS_VEP"]
    )
    GPN_MSA_SCORES  = os.path.expanduser(
        output_files["GPN_MSA_SCORES"]
    )
    OUT_GPN_MSA = os.path.expanduser(
        output_files["OUT_GPN_MSA"]
    )
    OUT_PRE_CADD = os.path.expanduser(
        output_files["OUT_PRE_CADD"]
    )
    OUT_CADD = os.path.expanduser(
        output_files["OUT_CADD"]
    )
    OUT_CADD_NA = os.path.expanduser(
        output_files["OUT_CADD_NA"]
    )

    logger.info(f"SHARDS_DIR: {SHARDS_DIR}")
    logger.info(f"BLOSUM_PATH: {BLOSUM_PATH}")
    logger.info(f"FASTA_PATH: {FASTA_PATH}")
    logger.info(f"GTF_PATH: {GTF_PATH}")
    logger.info(f"CADD_PATH: {CADD_PATH}")
    logger.info(f"OUT_CONCAT_ANNOTATIONS: {OUT_CONCAT_ANNOTATIONS}")
    logger.info(f"OUT_VAR_METADATA: {OUT_VAR_METADATA}")
    logger.info(f"OUT_PROCESS_VEP: {OUT_PROCESS_VEP}")
    logger.info(f"GPN_MSA_SCORES: {GPN_MSA_SCORES}")
    logger.info(f"OUT_GPN_MSA: {OUT_GPN_MSA}")
    logger.info(f"OUT_CADD: {OUT_CADD}")
    concat_annotations(SHARDS_DIR, OUT_CONCAT_ANNOTATIONS)
    write_variant_metadata(OUT_CONCAT_ANNOTATIONS, OUT_VAR_METADATA)
    process_vep(
        OUT_CONCAT_ANNOTATIONS,
        OUT_VAR_METADATA,
        FASTA_PATH,
        GTF_PATH,
        BLOSUM_PATH,
        OUT_PROCESS_VEP
    )
    # YET TO DO
    merge_gpn_msa(OUT_VAR_METADATA, GPN_MSA_SCORES, OUT_GPN_MSA)
    merge_all_pre_cadd(OUT_PROCESS_VEP, OUT_GPN_MSA, OUT_PRE_CADD)
    merge_cadd(nocadd_path  = OUT_PRE_CADD,
            cadd_file    = CADD_PATH,
            annotation_columns = config_general["annotation_columns"],
            output_path  = OUT_CADD)
            
    fill_nulls(
        input_path         = OUT_CADD,
        annotation_specs = config_general["annotation_specs"],
        cols_to_keep       = config_general["annotation_columns"],
        output_path        = OUT_CADD_NA,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--config_general",
        default="config_general.yaml",
        help="Path to config_general YAML file",
    )
    args = parser.parse_args()

    main(args.config, args.config_general)