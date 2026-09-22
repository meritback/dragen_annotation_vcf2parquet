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

# Fixed casts only, gnomAD columns are added per frame inside the function
BASE_CASTS = {
    "POS": pl.Int32,
    "QUAL": pl.Float32,
    "AC": pl.Int32,
    "AN": pl.Int32,
    "AF": pl.Float32,
    "distance": pl.Int32,
    # "strand": pl.Int8,
    "tsl": pl.Int8,
    "existing_inframe_oorfs": pl.Int16,
    "existing_outofframe_oorfs": pl.Int16,
    "existing_uorfs": pl.Int16,
    **{
        c: pl.Float32
        for c in [
            "af_2",
            "afr_af",
            "amr_af",
            "eas_af",
            "eur_af",
            "sas_af",
            "max_af",
            "cadd_phred",
            "cadd_raw",
            "revel",
        ]
    },
    **{f"spliceai_pred_ds_{s}": pl.Float32 for s in SPLICE},
    **{f"spliceai_pred_dp_{s}": pl.Int16 for s in SPLICE},
}


ANN_COLUMNS = [
    "ANN",
    "annotation",
    "annotation_impact",
    "gene_name",
    "gene_id",
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

VEP_TO_DROP = [
    # "clinvar",
    "clinvar_clndn",
    "clinvar_clndnincl",
    "clinvar_clndisdb",
    "clinvar_clndisdbincl",
    "clinvar_clnhgvs",
    "clinvar_clnrevstat",
    "clinvar_clnsig",
    "clinvar_clnsigconf",
    "clinvar_clnsigincl",
    "clinvar_clnvc",
    "clinvar_clnvcso",
    "clinvar_clnvi",
    "gerp",
    "QUAL",
    "FILTER",
    "AN",
    "impact",
    "symbol",
    "exon",
    "intron",
    "hgvsc",
    "hgvsp",
    "cdna_position",
    "existing_variation",
    "flags",
    "variant_class",
    "symbol_source",
    "hgnc_id",
    "mane_plus_clinical",
    "ensp",
    "swissprot",
    "trembl",
    "uniparc",
    "uniprot_isoform",
    "gene_pheno",
    "domains",
    "hgvs_offset",
    "af_2",
    "afr_af",
    "amr_af",
    "eas_af",
    "eur_af",
    "sas_af",
    "clin_sig",
    "somatic",
    "pheno",
    "pubmed",
    "overlapbp",
    "overlappc",
    "motif_name",
    "motif_pos",
    "high_inf_pos",
    "motif_score_change",
    "transcription_factors",
    "revel",
    "nmd",
    "spliceai_pred_dp_ag",
    "spliceai_pred_dp_al",
    "spliceai_pred_dp_dg",
    "spliceai_pred_dp_dl",
    "spliceai_pred_symbol",
    "spliceregion",
    "lof_filter",
    "lof_flags",
    "lof_info",
    "am_class",
    "mechpredict_pdn",
    "mechpredict_pgof",
    "mechpredict_plof",
    "mechpredict_prediction",
]

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
 
def _gtf_region_features(gtf: pl.DataFrame, region="transcript", biotypes=["protein_coding"]) -> pl.DataFrame:
    """Per-gene TSS, length and name from GTF 'gene' rows (protein-coding only).
 
    Returns columns: region, gene_name, gene_length, _tss, _gene_strand.
    """
    return (
        gtf.filter(pl.col("feature") == region)
        .with_columns(
            region=pl.col("attributes").str.extract(rf'{region}_id "([^"]+)"')
                  .str.split(".").list.first(),
            # transcript_name=pl.col("attributes").str.extract(rf'transcript_name "([^"]+)"'),
            # gene_name=pl.col("attributes").str.extract(rf'gene_name "([^"]+)"'),
            gene_type=pl.col("attributes")
                  .str.extract(r'gene_(?:type|biotype) "([^"]+)"'),
        )
        .filter(pl.col("gene_type").is_in(biotypes))
        .with_columns(
            region_length=pl.col("end") - pl.col("start") + 1,
            tss=pl.when(pl.col("strand") == "+")
                  .then(pl.col("start"))
                  .otherwise(pl.col("end")),
            Strand=pl.col("strand"),
        )
        .select(["region", "region_length", "tss", "Strand" ])
        .unique(subset=["region"])
    )

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

def _gtf_transcript_cds_length(gtf: pl.DataFrame) -> pl.DataFrame:
    """Sum CDS exon lengths per transcript.
 
    Same as add_more_annotations.py's gene_cds_len (CDS rows, end - start + 1,
    summed), but grouped by transcript_id instead of gene_id: VEP output has
    one row per transcript, and a gene-level sum counts exons shared between
    isoforms once per isoform.
 
    Returns: _tx (ENST without version), cds_length.
    """
    per_chrom = (
        gtf.filter(pl.col("feature").is_in(["CDS", "stop_codon"]))
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
        .agg(
            cds_length=pl.col("seg_len").filter(pl.col("feature") == "CDS").sum(),
            cds_length_incl_stop=pl.col("seg_len").sum(),
        )
    )
    # Guard for PAR genes annotated on both chrX and chrY under one ID:
    # keep one copy (non-chrY) instead of summing both.
    return (
        per_chrom.sort(["_tx", pl.col("chrom") == "chrY"])
        .unique(subset=["_tx"], keep="first", maintain_order=True)
        .select(["_tx", "cds_length", "cds_length_incl_stop"])
    )

# ---- RECREATION of TRANSCRIPT ORDER
"""
Re-create VEP's `--per_gene --pick_order ...` selection after the fact, on a
polars table with one row per VEP consequence (CSQ entry).

Mirrors ensembl-vep release/115 OutputFactory.pm:
  pick_VariationFeatureOverlapAllele_per_gene  -> group transcript rows by gene
  pick_worst_VariationFeatureOverlapAllele     -> rank each candidate per criterion
                                                  (0 = best), go through pick_order,
                                                  keep only the best at each step,
                                                  remaining ties -> first in VEP order
Non-transcript rows (regulatory, motif, intergenic) are kept untouched, as VEP does.
"""

PICK_ORDER = ["biotype", "mane_select", "canonical", "appris", "tsl",
              "ccds", "rank", "length", "ensembl", "refseq"]

# Consequence ranks from ensembl-variation release/115 Utils/Constants.pm
# (1..41, no ties; lower = more severe).
SO_RANK = {term: i for i, term in enumerate([
    "transcript_ablation", "splice_acceptor_variant", "splice_donor_variant",
    "stop_gained", "frameshift_variant", "stop_lost", "start_lost",
    "transcript_amplification", "feature_elongation", "feature_truncation",
    "inframe_insertion", "inframe_deletion", "missense_variant",
    "protein_altering_variant", "splice_donor_5th_base_variant",
    "splice_region_variant", "splice_donor_region_variant",
    "splice_polypyrimidine_tract_variant", "incomplete_terminal_codon_variant",
    "start_retained_variant", "stop_retained_variant", "synonymous_variant",
    "coding_sequence_variant", "mature_miRNA_variant", "5_prime_UTR_variant",
    "3_prime_UTR_variant", "non_coding_transcript_exon_variant", "intron_variant",
    "NMD_transcript_variant", "non_coding_transcript_variant",
    "coding_transcript_variant", "upstream_gene_variant", "downstream_gene_variant",
    "TFBS_ablation", "TFBS_amplification", "TF_binding_site_variant",
    "regulatory_region_ablation", "regulatory_region_amplification",
    "regulatory_region_variant", "intergenic_variant", "sequence_variant",
], start=1)}


def _blank(col):
    return pl.col(col).is_null() | pl.col(col).is_in(["", "-"])


def _flag(cond):
    """VEP convention: 0 if the transcript has the property, else 1."""
    return pl.when(cond.fill_null(False)).then(0).otherwise(1)


# Same values VEP computes internally (defaults for "missing" included).
CRITERIA = {
    "biotype":     _flag(pl.col("biotype") == "protein_coding"),
    "mane_select": _flag(~_blank("mane_select")
                         | pl.col("mane").str.contains("MANE_Select")),
    "canonical":   _flag(pl.col("canonical") == "YES"),
    # principal N -> N, alternative N -> N + 10, none -> 100  (output shows P1, A2, ...)
    "appris": (pl.col("appris").str.extract(r"(\d+)").cast(pl.Int32, strict=False)
               + pl.when(pl.col("appris").str.starts_with("A")).then(10).otherwise(0)
               ).fill_null(100),
    "tsl":    pl.col("tsl").str.extract(r"(\d+)").cast(pl.Int32, strict=False).fill_null(100),
    "ccds":   _flag(~_blank("ccds")),
    # most severe consequence of this transcript
    "rank":   pl.col("consequence").str.split("&")
                .list.eval(pl.element().replace_strict(SO_RANK, default=1000,
                                                       return_dtype=pl.Int32))
                .list.min().fill_null(1000),
    # only informative with --merged (SOURCE column); otherwise a tie, as in VEP
    "ensembl": _flag(pl.col("source").str.to_lowercase() == "ensembl"),
    "refseq":  _flag(pl.col("source").str.to_lowercase() == "refseq"),
}

def vep_per_gene(lf, length_col="cds_length", pick_order=PICK_ORDER,
                 variant_cols=("CHROM", "POS", "REF", "ALT")):
    """
    lf           : (Lazy)DataFrame, one row per VEP consequence entry
    length_col   : your transcript length column (longer = preferred).
                   VEP uses CDS length incl. stop codon for transcripts with a
                   translation, spliced transcript length otherwise.
    pick_order   : same criteria names as VEP's --pick_order
    variant_cols : columns identifying one VEP input variant
    """
    lf = pl.LazyFrame(lf) if isinstance(lf, pl.DataFrame) else lf
    # VEP fields can arrive as ints (e.g. tsl=1) or all-null; the criteria expect strings
    str_cols = ["biotype", "mane_select", "mane", "canonical", "appris",
                "tsl", "ccds", "consequence", "source"]
    schema = lf.collect_schema()
    lf = lf.with_columns(pl.col(c).cast(pl.Utf8) for c in str_cols if c in schema)
    lf = lf.with_row_index("_row")          # original (VEP) order = final tie-break

    crit = dict(CRITERIA)
    # VEP stores -length so that lower = better; missing -> 0 (as VEP's default)
    crit["length"] = -(pl.col(length_col).cast(pl.Int64, strict=False).fill_null(0))

    keys = [f"_pick_{c}" for c in pick_order]
    lf = lf.with_columns(**{f"_pick_{c}": crit[c] for c in pick_order})

    # lexicographic minimum over the criteria == VEP's category-by-category elimination
    group = [*variant_cols, "gene"]
    picked = pl.col("feature").sort_by([*keys, "_row"]).first().over(group)

    return (
        lf.filter(pl.col("feature_type").ne_missing("Transcript")
                  | (pl.col("feature") == picked))
          .sort("_row")
          .drop("_row", *keys)
    )

# ------------- 1. concatenate all shards of annotations into a single DataFrame and write to a parquet file
def concat_annotations(shards_dir: list[str], out_file: str, gene_filters=None):
    """Concatenate shards of annotations into a single parquet file.

    Gene / biotype filters are applied here, and the gnomAD population AFs are
    collapsed into a single `maf_gnomad` column (max across populations) before
    all raw gnomadg_* columns are dropped, so the concatenated file only holds
    what the downstream steps need.
    """
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
    # Shards keep their original column casing (process_vep lowercases later),
    # so look up the columns used here case-insensitively.
    lower_to_name = {c.lower(): c for c in names}

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
            lf = lf.filter(pl.col(lower_to_name["gene"]).is_in(genes_to_keep))

        biotypes = gene_filters.get("biotypes", None)
        if biotypes is not None:
            logger.info(f"Filtering for biotypes: {biotypes}")
            lf = lf.filter(pl.col(lower_to_name["biotype"]).is_in(biotypes))

    # ── gnomAD: keep only the max AF across populations ───────────────────
    gnomad_cols = [c for c in names if c.lower().startswith("gnomadg_")]
    gnomad_af_cols = [c for c in gnomad_cols if c.lower().startswith("gnomadg_af_")]
    if not gnomad_af_cols:
        raise ValueError("No gnomadg_af_* columns found, cannot compute maf_gnomad")
    logger.info(
        f"Computing maf_gnomad from {len(gnomad_af_cols)} gnomAD AF columns, "
        f"dropping {len(gnomad_cols)} gnomadg_* columns"
    )

    casts = {
        **{c: t for c, t in BASE_CASTS.items() if c in names},
        **{c: pl.Float32 for c in gnomad_af_cols},
    }

    # Row-wise steps (length filter, casts, maf_gnomad, gnomAD drop) run before
    # the unique so it only has to hold the slimmed-down table.
    (
        lf.filter(
            (
                pl.col("ALT").str.len_chars().cast(pl.Int64)
                - pl.col("REF").str.len_chars().cast(pl.Int64)
            ).abs() < 50
        )
        .with_columns(
            [pl.col(c).cast(t, strict=False) for c, t in casts.items()]
        )
        .with_columns(maf_gnomad=pl.max_horizontal(gnomad_af_cols))
        .drop(gnomad_cols)
        .unique(subset=["ID", "feature"])
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
    region="gene", # can be transcript or gene
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

    # Gene / biotype filters are applied in concat_annotations. biotypes is
    # still needed below to subset the GTF for the distance-to-TSS step.
    biotypes = (gene_filters or {}).get("biotypes", None)

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
    vep_file = vep_file.unique(
        subset=["id", "feature"]
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

    # ---- is_plof ------------- from #https://github.com/HolEv/deeprvat-rd/blob/790853f8113311cb03bc99a5198004cf96d756d6/solve_rd_preprocessing/6_deeprvat_variant_scores.ipynb#L638
    PLOF_COLS=[
        "consequence_stop_gained",
        "consequence_frameshift_variant",
        "consequence_stop_lost",
        "consequence_start_lost",
        "consequence_splice_acceptor_variant",
        "consequence_splice_donor_variant",
    ]
    annos = annos.with_columns(is_plof=pl.any_horizontal([pl.col(c) for c in PLOF_COLS]).cast(pl.Int8))

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
    # maf_gnomad is computed in concat_annotations (raw gnomadg_* already dropped)

    tmp = Path(output_path).with_suffix(".stage1.parquet")
    annos.sink_parquet(tmp, engine="streaming")
    annos = pl.scan_parquet(tmp)

    #-------- was not run with --total_length, so cds_position is just the start (or range) of the variant in the CDS
        # ── GENCODE v49 GTF (VEP 115) ─────────────────────────────────────────
    logger.info(f"Loading GTF {gtf_path}")
    gtf = _load_gtf_polars(gtf_path)
    genes_df = _gtf_region_features(gtf, region="gene")
    transcripts_df = _gtf_region_features(gtf, region="transcript")
    tx_cds_len = _gtf_transcript_cds_length(gtf)
    del gtf
    gc.collect()
    logger.info(
        f"  GTF: {genes_df.height} protein-coding genes, "
        f"{transcripts_df.height} transcripts, "
        f"{tx_cds_len.height} transcripts with CDS"
    )

    # ── relative_cds_position: cds_start / total CDS length ─────────────────────
    # As in add_more_annotations.py, but the CDS length is per transcript
    # (joined on 'feature'), because there is one row per transcript.
    logger.info("Computing relative CDS positions (cds_start / GTF CDS length)")
    annos = (
        _idempotent_join(
            annos.with_columns(_tx=_tx_key()),
            tx_cds_len.lazy(), on=["_tx"],validate="m:1",
        )
        .with_columns(
            relative_cds_position=(
                _cds_start_expr().cast(pl.Float64)
                / pl.col("cds_length_incl_stop").cast(pl.Float64)
            )
            .clip(0.0, 1.0).round(4).cast(pl.Float32)
        )
        .drop("_tx")
        # keep cds_length – also used by next in frame
    )
    match_rate = (
        annos.filter(pl.col("cds_position").is_not_null())
        .select(pl.col("cds_length_incl_stop").is_not_null().mean())
        .collect()
        .item()
    )
    logger.info(f"  CDS length found for {match_rate} of rows with cds_position")

    # ── next_in_frame_relative (start_lost SNVs only) ───────────────────────────
    logger.info("Processing start_lost variants for next in frame ATG")
    annos = _compute_next_in_frame(annos, tx_cds_len, fasta_path)

    #NOTE: new: if region is gene: now filter
    if region == "gene":
        logger.info("Filtering to one transcript per gene (VEP pick_order)")
        annos = vep_per_gene(annos, length_col="cds_length_incl_stop", variant_cols=["chrom", "pos", "ref", "alt"])

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

    #── 5' UTR dummies ────────────────────────────────────────────────────
    logger.info("Processing 5' UTR variant consequence annotations")
    C = "five_prime_utr_variant_consequence"
    if "5utr_consequence" in annos.collect_schema().names():
        annos = annos.with_row_index("_row")
        dummies = (
            annos.select("_row", pl.col("5utr_consequence").str.split("&").alias(C))
            .explode(C)
            .filter(pl.col(C).is_not_null() & (pl.col(C) != ""))
            .collect()
            .to_dummies(columns=C)
            .group_by("_row").max()
        )
        annos = annos.join(dummies.lazy(), on="_row", how="left", validate="1:1").drop("_row")

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

    # ── Distance to TSS ───────────────────────────────────────────────────
    annos = annos.with_columns(region=pl.col("feature"))
    logger.info("Joining distance to transcript-specific TSS")
    anno_tss = (
        annos.select(["id", "pos", "region"])
        .join(transcripts_df.lazy(), on="region", how="left")
        .with_columns(
            dist_to_tss=pl.when(pl.col("Strand") == "+")
            .then(pl.col("pos") - pl.col("tss"))
            .otherwise(pl.col("tss") - pl.col("pos"))
        )
        # .collect(engine="streaming")
    )
    anno_tss = anno_tss.rename({col: col.lower() for col in anno_tss.collect_schema().names()})
    annos = annos.join(anno_tss.lazy(), on=["id", "pos", "region", "strand"], how="left")

    # ── Set region = transcript/feature (transcript-specific TSS), can be gene if chosen ─────────
    if region=="transcript":
        logger.info("choosing transcript-specific TSS (region=feature)")
        annos = annos.with_columns(region=pl.col("feature"))
    elif region=="gene":
        logger.info("choosing gene-specific TSS (region=gene)")
        annos = annos.with_columns(region=pl.col(region))
    else:
        raise ValueError(f"Invalid region: {region}. Must be 'gene' or 'transcript'.")

    logger.info(f"Writing VEP-processed annotations to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    annos.sink_parquet(output_path, engine="streaming")
    logger.info("VEP processing complete")
    tmp.unlink(missing_ok=True)



# ── Step 2b: GPN-MSA ──────────────────────────────────────────────────────

def merge_gpn_msa(variant_metadata_path, scores_gpn_msa_file, output_path):
    """Merge GPN-MSA scores for all chromosomes (one job)."""
    logger.info("Loading annotations for GPN-MSA")
    vm = pl.read_parquet(variant_metadata_path, columns=["id", "chrom"])

    scores_lazy = pl.scan_parquet(scores_gpn_msa_file)
    unique_chroms = vm["chrom"].unique().to_list()
    logger.info(f"Processing GPN-MSA for {len(unique_chroms)} chromosomes")

    all_scores = []
    for chrom in unique_chroms:
        gpn_chrom = chrom.lstrip("chr")
        scores_chr = (
            scores_lazy.filter(pl.col("chrom").cast(pl.String).str.to_lowercase() == gpn_chrom.lower())
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
        sampled_chrom = vm.filter(pl.col("chrom").cast(pl.String).str.to_lowercase() == chrom.lower()).select("id").lazy()
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
    gpn_msa_path,
    output_path,
):
    """
    Merge all external scores (except CADD) into the main annotation table:
      AbSplice2, GPN-MSA, AbExp, PromoterAI, PrimateAI3D,
      protein domains (MobiDB/TED/LC), CPT-1, ENCODE.
    Writes to output_path (the pre-CADD intermediate).
    """
    logger.info("Loading VEP-processed annotations")
    
    annos = (
        pl.read_parquet(vep_processed_path)
        .sort("dist_to_tss", nulls_last=True)
        .unique(subset=["id", "feature"], keep="first")
        .lazy()
    )

    # ── GPN-MSA ───────────────────────────────────────────────────────────
    logger.info("Merging GPN-MSA scores")
    gpn_msa = pl.scan_parquet(gpn_msa_path)
    annos = annos.join(gpn_msa, on="id", how="left", validate="m:1")

    logger.info(f"Writing pre-CADD annotations to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    annos.sink_parquet(output_path, engine="streaming")
    logger.info("Pre-CADD merge complete")


# ── Step 4: Fill nulls ──────────────────────────────────────────────────────


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
    annos = annos.filter(pl.col("maf_gnomad")<0.001)
    annos.sink_parquet(output_path.replace(".parquet", "_maf001.parquet"), engine="streaming")
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
    GPN_MSA_SCORES = os.path.expanduser(
        output_files["GPN_MSA_SCORES"]
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
    logger.info(f"OUT_PRE_CADD: {OUT_PRE_CADD}")
    logger.info(f"OUT_CADD: {OUT_CADD}")
    logger.info(f"OUT_CADD_NA: {OUT_CADD_NA}")

    concat_annotations(
        SHARDS_DIR, OUT_CONCAT_ANNOTATIONS, config_general["gene_filters"]
    )
    write_variant_metadata(OUT_CONCAT_ANNOTATIONS, OUT_VAR_METADATA)
    
    process_vep(
        annotation_file=OUT_CONCAT_ANNOTATIONS,
        variant_metadata_file=OUT_VAR_METADATA,
        fasta_path=FASTA_PATH,
        gtf_path=GTF_PATH,
        blosum_path=BLOSUM_PATH,
        output_path=OUT_PROCESS_VEP,
        gene_filters=config_general["gene_filters"],
        region=config_general["region"],
    )
    # YET TO DO
    merge_gpn_msa(OUT_VAR_METADATA, GPN_MSA_SCORES, OUT_GPN_MSA)
    merge_all_pre_cadd(OUT_PROCESS_VEP, OUT_GPN_MSA, OUT_PRE_CADD)
    fill_nulls(
        input_path         = OUT_PRE_CADD,
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
        help="Path to general config YAML file",
    )
    args = parser.parse_args()

    main(args.config, args.config_general)