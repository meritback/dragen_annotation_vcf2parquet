# aggv3-annotations

AggV3 VEP-annotated msVCF → long-format annotation parquet.

One row per variant per consequence (i.e. per transcript), per subshard: the output joins to that pipeline's `merged.parquet` on
`CHROM`/`POS`/`REF`/`ALT`.

---

## What it does

| Source | File | Contributes |
|---|---|---|
| Annotated msVCF | `annotation_root/shard-*/subshard-*/dragen.gel.annotated.vcf.gz` | `CSQ` expanded to ~200 columns, `ANN` raw, site `AC/AN/AF`, `QUAL`, `FILTER` |

```
subshard ──► CONVERT_ANNOTATIONS ──► annotations.parquet + columns.txt + convert.log
                      │
                      └──► schema_hash.tsv ──► CHECK_SCHEMA ──► schema.tsv
```

Two processes, not five. Unlike the genotype pipeline there is nothing to join: each
subshard's annotated VCF is self-contained and the extraction is a single bcftools pass.
The work is embarrassingly parallel across ~3100 subshards.

### Why this is much lighter than the genotype pipeline

There is no sample dimension. A genotype row is per sample per variant across 138k samples;
an annotation row is per variant per consequence. The ~200 CSQ columns look alarming but
they are one variant's worth of strings, not 138k — so `process_medium` (2 CPU / 8 GB) covers
the only real process here, against the genotype pipeline's 32 GB.

### Why CHECK_SCHEMA exists

The column list is derived from **each VCF's own CSQ header**. If the annotation release is
not internally consistent — a shard re-run against a different VEP cache, a plugin added
partway through — the parquets get different schemas and will not concatenate. Without a
check that surfaces weeks later, on a partial read, at the point of use.

Each task emits an md5 of its column list as one short line; `collectFile` concatenates them
and a single tiny task compares. It does not stage 3100 column files — on awsbatch every
staged input is a real download.

---

## Quick start

```bash
# pilot: one subshard, and read report.html before going further
nextflow run . -profile test,docker \
    --annotation_root /path/to/functional-annotation_2025-12-24 \
    --outdir pilot

# full run
nextflow run . -profile docker \
    --annotation_root /path/to/functional-annotation_2025-12-24 \
    --outdir results

# interactive session, 4 CPU / 30 GB
nextflow run . -profile local_small,conda \
    --annotation_root /path/to/functional-annotation_2025-12-24 \
    --outdir results

nextflow run . --help
```

Run under `tmux` or `screen`. 3100 subshards is a long job and a dropped connection
otherwise kills it — though `-resume` will pick it back up.

### Repository layout

```
main.nf                      workflow
nextflow.config              resources, profiles, retry policy
nextflow_schema.json         parameter documentation / CloudOS form
environment.yml              conda spec (python 3.12, bcftools 1.21, polars 1.x, pyarrow, click)
bin/1_vcf2annotations.py     VCF → parquet, plus a `list_fields` subcommand
```

---

## Output

```
outdir/shard-{S}/subshard-{U}/annotations.parquet
                             /columns.txt
                             /convert.log
outdir/pipeline_info/schema.tsv          canonical column list, index → name
                    /schema_hashes.tsv   one line per subshard
                    /schema_check.log
                    /trace.txt, report.html, timeline.html
```

### Columns

**Fixed, uppercase:** `CHROM` `POS` `REF` `ALT` `QUAL` `FILTER` `AC` `AN` `AF`, then `ANN`,
then an `ID` column (`CHROM:POS:REF:ALT`) appended last.

Uppercase and typed to match the genotype parquets — `CHROM` is the same `pl.Enum`, `POS` the
same `Int64` — because these four are the join keys and polars will refuse or silently upcast
a mismatch.

**Expanded CSQ, lowercase snake_case:** ~200 columns, sanitised from the VEP header
(`cDNA_position` → `cdna_position`, `MAX_AF_POPS` → `max_af_pops`). Names that collide after
sanitisation get a `_2`, `_3` suffix rather than being dropped — losing one would shift every
column after it.

**`ANN`** is SnpEff's block, carried whole and unparsed, repeated on each CSQ row for that
variant. `split-vep` expands one field per invocation, and the two are per-transcript lists of
different lengths with no correct pairing, so there is no way to explode both at once. Nothing
is lost; split it downstream if you need it, and ZSTD compresses the repetition to near
nothing. `--annotation_field ANN` swaps which one is expanded.

### Types

Only the fixed columns are cast. **The CSQ subfields stay `Utf8` on purpose**: many are
multi-valued (`&`-separated), and VEP writes `""`, `.` and `-` for missing in different fields,
so a `strict=False` cast would silently null every one of those. Cast at query time, where you
can see what you lost:

```python
pl.col("cadd_phred").cast(pl.Float64, strict=False)
pl.col("gnomadg_af_joint").cast(pl.Float64, strict=False)
```

```sql
-- duckdb
SELECT * FROM 'outdir/shard-*/subshard-*/annotations.parquet'
WHERE TRY_CAST(af AS DOUBLE) < 0.01
```

---

## Notes and gotchas

**`-a CSQ` is pinned in both bcftools calls.** These VCFs carry both `CSQ` and `ANN`, and
without `-a` split-vep auto-detects in the order CSQ, BCSQ, ANN and warns. The script calls
bcftools twice — once for the header, once for the data — and if those resolved to different
fields, the column names and the values would silently disagree. `BCSQ` is not present in this
release despite what the warning implies; it lists all three regardless.

**Positional alignment is load-bearing.** `-A tab` emits subfields in header order and nothing
downstream re-checks that the Nth name describes the Nth column. `quote_char=None` on the CSV
read is part of this: left on, a stray double quote in an HGVS or domain string would swallow
tabs and newlines until the next one and misalign everything after it.

**`-x` drops sites with no consequence at all**, so the parquet is not a complete record of the
VCF. Measure the gap if it matters:
`bcftools view -H -i 'INFO/CSQ="."' "$VCF" | wc -l`

**`--max_af` is usually not worth setting.** At ~138k samples the site frequency spectrum is
dominated by singletons, so AF<0.01 retains roughly 97-99% of sites. Filtering at query time
costs almost no disk and keeps the threshold changeable without re-running 3100 subshards.

**Multi-allelic sites.** `AF` is `Number=A`, so a site with two ALTs writes `0.5,0.001` into a
single cell and any numeric cast returns null. Check before relying on the column:
`bcftools view -H "$VCF" | awk '$5 ~ /,/' | wc -l`. If nonzero, normalise with
`bcftools norm -m -any` upstream.

**bcftools stderr goes to `convert.log`**, published per subshard, not to `/dev/null`. Stale
`.tbi` warnings and subshards that quietly produced few rows are only diagnosable if it is kept.

**`%ID` is deliberately not extracted** — the converter appends its own `ID` column and the
names would collide. rsIDs are in the CSQ `Existing_variation` subfield anyway.
