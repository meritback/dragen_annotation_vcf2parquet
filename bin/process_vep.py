import gc
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import polars as pl
import pyranges as pr
from pathlib import Path
from tqdm import tqdm

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
def concat_annotations(shards_dir: List[str], out_file: str):
    """Concatenate shards of annotations into a single DataFrame and write to a parquet file."""
    # there can be multiple shards folderes f (value in list shards_dir), each containing the annotation files like do:
    # {f}/shard-{*}/subshard-{*}/annotations.parquet
    # all_shards should be a list of the paths to   all the parquet files in all the shards folders
    all_shards = []
    for f in shards_dir:
        all_shards.extend(glob.glob(os.path.join(f, "shard-*/subshard-*/annotations.parquet")))
    df = pl.scan_parquet(all_shards).collect()
    df.write_parquet(out_file)

# ------------- 2. write variant metadata to a parquet file
def write_variant_metadata(annotations_file: str, out_file: str):
    # Load the annotations file
    annos = pl.read_parquet(annotations_file)
    annos = annos.rename({col: col.lower() for col in annos.columns})

    # Create a DataFrame with the variant metadata
    vm = annos.select(["id", "chrom", "pos", "ref", "alt"]).unique(subset=["id", "chrom", "pos", "ref", "alt"])

    # Write the variant metadata to a parquet file
    vm.write_parquet(out_file)

# ------------- 3. process VEP annotations
def process_vep(
    annotation_file,
    # variant_metadata_file,
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
        .with_columns(pl.col("consequence").str.split(","))
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
    vep_file = vep_file.unique()

    # ── Load variant metadata ──────────────────────────────────────────────
    # logger.info("Loading variant metadata")
    # variant_metadata = pl.scan_parquet(variant_metadata_file)

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
    annos = annos.with_columns("AF").alias("maf_cohort")
    # vm_schema = variant_metadata.collect_schema().names()
    # if "mac_cohort" in vm_schema and n_samples is not None:
    #     logger.info(f"{n_samples} samples, computing MAF from MAC")
    #     cols_to_select = [c for c in vm_schema if "cohort" in c]
    #     maf_df = variant_metadata.select(["id"] + cols_to_select).with_columns(
    #         maf_cohort=pl.col("mac_cohort") / (2 * n_samples)
    #     )
    #     annos = annos.join(maf_df, on="id", how="left", validate="m:1")

    # ── Relative CDS position ─────────────────────────────────────────────
    logger.info("Computing relative CDS positions")
    sites = (
        annos.with_columns(cds_parts=pl.col("cds_position").str.split("/"))
        .with_columns(
            length=pl.col("cds_parts").list.get(1),
            protein_pos=pl.col("cds_parts").list.get(0),
        )
        .with_columns(
            pl.col("protein_pos")
            .map_elements(convert_to_int_and_get_max, return_dtype=pl.Int64)
            .alias("protein_pos")
        )
        .filter(pl.col("protein_pos").is_not_null())
        .with_columns(pl.col("length").cast(pl.Int64))
        .with_columns(
            (pl.col("protein_pos") / pl.col("length"))
            .round(2)
            .alias("relative_cds_position")
        )
    )
    cds_merged = annos.select("id", "gene").join(
        sites.select("id", "gene", "relative_cds_position"),
        on=["id", "gene"],
        how="left",
    )
    annos = annos.join(cds_merged, on=["id", "gene"], how="left", validate="1:1")

    # ── Start-lost: next in-frame ATG ─────────────────────────────────────
    logger.info("Processing start_lost variants for next in-frame ATG")
    schema_names = annos.collect_schema().names()
    if "consequence_start_lost" in schema_names:
        vep_start_lost = (
            annos.filter(pl.col("consequence_start_lost") == 1)
            .select(
                [
                    "pos",
                    "chrom",
                    "gene",
                    "id",
                    "cds_position",
                    "codons",
                    "strand",
                    "allele",
                ]
            )
            .collect()
            .to_pandas()
        )
        vep_start_lost_snv = vep_start_lost[
            vep_start_lost["allele"].str.len() == 1
        ].copy()
        vep_start_lost_snv[["variant_pos_cds", "cds_length"]] = vep_start_lost_snv[
            "cds_position"
        ].str.split("/", expand=True)
        vep_start_lost_snv = vep_start_lost_snv[
            vep_start_lost_snv["variant_pos_cds"].str.len() == 1
        ]
        vep_start_lost_snv["variant_pos_cds"] = vep_start_lost_snv[
            "variant_pos_cds"
        ].astype(int)
        vep_start_lost_snv["cds_length"] = vep_start_lost_snv["cds_length"].astype(int)
        vep_start_lost_snv["Chromosome"] = vep_start_lost_snv["chrom"].str.replace(
            r"^(?!chr)", "chr", regex=True
        )

        logger.info(f"Processing {len(vep_start_lost_snv)} start_lost SNVs")
        strands_pos = get_regions_positive_strand(vep_start_lost_snv, fasta_path)
        strands_neg = get_regions_negative_strand(vep_start_lost_snv, fasta_path)
        start_lost_df = pd.concat([strands_pos, strands_neg])
        start_lost_df["next_in_frame"] = start_lost_df.apply(
            lambda x: next_inframe_start_codon_distance(x["seq"], search_init=3), axis=1
        )
        start_lost_df["next_in_frame_relative"] = (
            start_lost_df["next_in_frame"] / start_lost_df["cds_length"]
        )
        start_lost_df.loc[
            start_lost_df["next_in_frame_relative"] < 0, "next_in_frame_relative"
        ] = 1
        start_lost_pl = pl.DataFrame(
            start_lost_df[["gene", "id", "next_in_frame_relative"]]
        )
        annos = annos.join(
            start_lost_pl.lazy(), on=["id", "gene"], how="left", validate="1:1"
        )

    # ── SpliceAI max delta score ───────────────────────────────────────────
    logger.info("Processing SpliceAI predictions")
    if "spliceai_pred" in annos.collect_schema().names():
        annos = annos.with_columns(
            pl.col("spliceai_pred")
            .str.split("|")
            .list.slice(1, 4)
            .list.eval(pl.element().cast(pl.Float32, strict=False))
            .list.max()
            .alias("spliceai_delta_score")
        ).drop("spliceai_pred")

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
    #         "id", "gene", "five_prime_utr_variant_consequence"
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
    #         utr_df.select("id", "gene", "row_nr")
    #         .join(dummies.lazy(), on="row_nr", how="left")
    #         .drop("row_nr")
    #     )
    #     annos = annos.join(utr_df, on=["id", "gene"], how="left", validate="1:1")

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

    logger.info(f"Writing VEP-processed annotations to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    annos.sink_parquet(output_path, engine="streaming")
    logger.info("VEP processing complete")



# ── Step 2b: GPN-MSA ──────────────────────────────────────────────────────

def merge_gpn_msa(annotation_file, scores_gpn_msa_file, output_path):
    """Merge GPN-MSA scores for all chromosomes (one job)."""
    logger.info("Loading annotations for GPN-MSA")
    annos = pl.scan_parquet(annotation_file)
    sampled = annos.select(["id", "chrom"]).unique(subset=["id", "chrom"])

    scores_lazy = pl.scan_parquet(scores_gpn_msa_file)
    unique_chroms = sampled["chrom"].unique().to_list()
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
        sampled_chrom = sampled.filter(pl.col("chrom").str.to_lowercase() == chrom.lower()).select("id").lazy()
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
    #     .select("promoterai", "promoterai_abs", "id", "gene")
    # )
    # scores = annos.select("gene", "id").join(
    #     promoter_ai, how="left", on=["id", "gene"], validate="1:m"
    # )
    # scores_grouped = scores.group_by(["gene", "id"]).agg(
    #     pl.col("promoterai").sort_by(pl.col("promoterai_abs"), descending=True).first()
    # )
    # annos = annos.join(
    #     scores_grouped.select("promoterai", "id", "gene"),
    #     how="left",
    #     on=["id", "gene"],
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
def main():
    # Paths
    SHARDS_DIR = ("shards1-10", "shards11-98")#something like this

    BLOSUM_PATH = "path/to/References/BLOSUM/blosum62.txt"
    FASTA_PATH = "path/to/References/GENCODE/GRCh38.primary_assembly.genome.fa"
    GTF_PATH = "path/to/References/GENCODE/gencode.v49.annotation.gtf"

    OUT_CONCAT_ANNOTATIONS = "Data_out/annotations/annotations_processed/annotations.parquet"
    OUT_VAR_METADATA = "Data_out/annotations/annotations_processed/variant_metadata.parquet"
    OUT_PROCESS_VEP = "Data_out/annotations/annotations_processed/annotations_vep_processed.parquet"



    concat_annotations(SHARDS_DIR, OUT_CONCAT_ANNOTATIONS)
    write_variant_metadata(OUT_CONCAT_ANNOTATIONS, OUT_VAR_METADATA)
    process_vep(
        OUT_CONCAT_ANNOTATIONS,
        FASTA_PATH,
        GTF_PATH,
        BLOSUM_PATH,
        OUT_PROCESS_VEP,
    )