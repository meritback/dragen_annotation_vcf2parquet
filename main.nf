#!/usr/bin/env nextflow
/*
 * AggV3 VEP-annotated msVCF -> long-format annotation parquet, one task per
 * subshard.
 *
 *   CONVERT_ANNOTATIONS  dragen.gel.annotated.vcf.gz -> annotations.parquet
 *   CHECK_SCHEMA         every subshard's column list -> schema.tsv
 *
 * Two processes, not five. Unlike the genotype pipeline there is nothing to
 * join here: each subshard's annotated VCF is self-contained, the extraction
 * is a single bcftools pass, and there is no sample dimension -- one row per
 * variant per consequence, not per sample per variant. The whole thing is
 * embarrassingly parallel across ~3100 subshards.
 *
 * CHECK_SCHEMA is the one piece of cross-subshard logic and it is worth its
 * cost. The column list is derived from each VCF's own CSQ header, so if the
 * annotation release is not internally consistent -- a shard re-run against a
 * different VEP cache, a plugin added partway through -- the parquets will
 * have different schemas and will not concatenate. That failure surfaces
 * weeks later at the point of use, on a partial read, unless something checks
 * it here. It is a single tiny task over N text files.
 */

nextflow.enable.dsl = 2

// ---------------- parameters ----------------
params.annotation_root = null       // .../filesystems/functional-annotation_2025-12-24
params.outdir          = 'results'
params.shards          = 'shard-*'  // 'shard-1' to pilot a single shard
params.subshards       = 'subshard-*'
params.vcf_name        = 'dragen.gel.annotated.vcf.gz'

// which INFO field to expand; the other one is carried as a raw string column
params.annotation_field = 'CSQ'

// site filters, applied in bcftools during extraction
params.max_af          = null
params.max_ac          = null

params.help            = false

// Split each selected subshard into N coordinate ranges of equal record count
// and publish them as subshard-11a, subshard-11b, ... Off by default; meant
// for the handful of subshards that are too large to do in one task --
// shard-20/subshard-11 is >0.9 GB compressed, several times the next largest,
// and the intermediate TSV scales with it.
//
//   --chunked true   4 chunks (the default when chunking is on)
//   --chunked 6      6 chunks
//   (absent)         one task per subshard, as before
params.chunked         = false

def helpMessage() {
    log.info """
    Usage:
      nextflow run . --annotation_root <dir> --outdir <dir>
      nextflow run . -profile test,docker --annotation_root ...

    Required:
      --annotation_root  Directory holding shard-*/subshard-*/${params.vcf_name}

    Optional:
      --shards            Glob for shards            [${params.shards}]
      --subshards         Glob for subshards         [${params.subshards}]
      --annotation_field  INFO field to expand       [${params.annotation_field}]
      --max_af            Drop sites above this AF   [${params.max_af}]
      --max_ac            Drop sites above this AC   [${params.max_ac}]
      --outdir            Publish directory          [${params.outdir}]
      --chunked           Split each subshard into N coordinate ranges and
                          publish them as subshard-Na, subshard-Nb, ...
                          'true' means 4; pass an integer for another count.
                          [${params.chunked}]

    Note on --chunked: for the few subshards too large to convert in one task.
      nextflow run . --annotation_root <dir> \\
          --shards shard-20 --subshards subshard-11 --chunked true

    Note on --max_af: at AggV3's cohort size (~138k) the site frequency
    spectrum is overwhelmingly singletons, so AF<0.01 removes only a few
    percent of sites. Filtering at query time costs almost no disk and keeps
    the threshold changeable without re-running 3100 subshards.
    """.stripIndent()
}

// ---------------- processes ----------------
process DIAGNOSE {
    label 'process_low'
    publishDir "${params.outdir}/pipeline_info", mode: 'copy'

    output:
    path 'diag.txt'

    script:
    '''
    set +eu

    {
      echo "== bcftools =="
      which bcftools
      bcftools --version 2>&1 | head -3

      echo "== BCFTOOLS_PLUGINS = [${BCFTOOLS_PLUGINS:-UNSET}] =="

      echo "== +split-vep -h exit code =="
      bcftools +split-vep -h > /dev/null 2>&1
      echo "rc=$?"

      echo "== +split-vep -h output =="
      bcftools +split-vep -h 2>&1 | head -15

      echo "== plugin -lv =="
      bcftools plugin -lv 2>&1 | head -25

      echo "== split-vep.so on disk =="
      find /opt /usr /srv -maxdepth 7 -name 'split-vep*' 2>/dev/null

      echo "== libexec listing =="
      ls -la /opt/conda/libexec/bcftools 2>&1 | head -20

      echo "== activate.d =="
      cat /opt/conda/etc/conda/activate.d/*bcftools* 2>&1 | head

      echo "== python / polars =="
      which python; python --version 2>&1
      python -c "import polars as pl; pl.show_versions()" 2>&1 | head -8
    } > diag.txt 2>&1

    cat diag.txt
    exit 0
    '''
}

process CONVERT_ANNOTATIONS {
    tag "${shard}/${subshard}"
    label 'process_medium'

    publishDir params.outdir,
        mode: 'copy',
        overwrite: false,
        saveAs: { fn -> fn == 'versions.yml' ? null : "${shard}/${subshard}/${fn}" }

    input:
    tuple val(shard), val(subshard), path(vcf), path(idx)

    output:
    tuple val(shard), val(subshard), path("annotations.parquet"), emit: parquet
    path "columns.txt",     emit: columns
    path "schema_hash.tsv", emit: hash
    path "convert.log",     emit: log
    path "versions.yml",    emit: versions

    script:
    def af_arg = params.max_af ? "--max-af ${params.max_af}" : ''
    def ac_arg = params.max_ac ? "--max-ac ${params.max_ac}" : ''
    """
    # Intermediate TSV stays in the task work dir: concurrent tasks never share
    # scratch. Same fix as the genotype pipeline's full-disk failures.
    export TMPDIR=\$PWD

    # bcftools stderr goes to a published log rather than /dev/null. A stale
    # .tbi warning or a subshard that silently produced few rows is only
    # diagnosable if this is kept.
    set -o pipefail
    vcf2annotations.py convert_annotations \\
        ${vcf} annotations.parquet \\
        --columns-file columns.txt \\
        --annotation-field ${params.annotation_field} \\
        ${af_arg} ${ac_arg} 2>&1 | tee convert.log

    # One short line per subshard instead of shipping 3100 column lists to a
    # single task. On awsbatch every staged input is a real download, so the
    # hash keeps CHECK_SCHEMA a genuinely cheap task.
    printf '%s\\t%s/%s\\t%s\\n' \\
        "\$(md5sum columns.txt | cut -d' ' -f1)" \\
        "${shard}" "${subshard}" "\$(wc -l < columns.txt)" > schema_hash.tsv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bcftools: \$(bcftools --version | head -1 | sed 's/^bcftools //')
        polars: \$(python -c "import polars; print(polars.__version__)")
    END_VERSIONS
    """

    stub:
    "touch annotations.parquet columns.txt convert.log versions.yml"
}

process CHECK_SCHEMA {
    label 'process_low'

    publishDir "${params.outdir}/pipeline_info", mode: 'copy', overwrite: true

    input:
    path hashes            // one line per subshard: md5, shard/subshard, ncols
    path 'reference.txt'   // any one subshard's column list

    output:
    path "schema.tsv"
    path "schema_check.log"
    path "schema_variants.tsv", optional: true

    script:
    """
    #!/usr/bin/env python3
    import collections, sys

    groups = collections.defaultdict(list)
    ncols = {}
    for line in open("${hashes}"):
        if not line.strip():
            continue
        h, name, n = line.rstrip("\\n").split("\\t")
        groups[h].append(name)
        ncols[h] = n

    ref = [c for c in open("reference.txt").read().splitlines() if c]
    with open("schema.tsv", "w") as out:
        for i, c in enumerate(ref):
            out.write(f"{i}\\t{c}\\n")

    lines = [
        f"subshards      : {sum(len(v) for v in groups.values())}",
        f"schema variants: {len(groups)}",
        f"columns        : {len(ref)}",
    ]

    if len(groups) > 1:
        # Ordered by size: the majority schema first, then the outliers with a
        # named example each, so the next step is to diff two concrete files.
        with open("schema_variants.tsv", "w") as out:
            for h, names in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                out.write(f"{h}\\t{len(names)}\\t{ncols[h]}\\t{names[0]}\\n")
        lines.append("")
        lines.append("Subshards do not share a column schema, so these "
                     "parquets will not concatenate. See schema_variants.tsv "
                     "for the group sizes and one example subshard each.")

    open("schema_check.log", "w").write("\\n".join(lines) + "\\n")
    print("\\n".join(lines))
    if len(groups) > 1:
        sys.exit(1)
    """

    stub:
    "touch schema.tsv schema_check.log"
}

process PARTITION_VCF {
    tag "${shard}/${subshard}"
    label 'process_low'

    publishDir "${params.outdir}/pipeline_info", mode: 'copy', overwrite: true,
        saveAs: { fn -> "${shard}_${subshard}_${fn}" }

    input:
    tuple val(shard), val(subshard), path(vcf), path(idx), val(nchunk)

    output:
    tuple val(shard), val(subshard), path(vcf), path(idx), path('regions.txt')

    script:
    """
    # CHROM/POS only: one full decompression pass, but nothing is held in
    # memory and the boundaries come out exact rather than guessed from the
    # coordinate span. Variant density is not uniform inside a subshard, so
    # equal spans would not give equal chunks.
    set -o pipefail
    bcftools query -f '%CHROM\\t%POS\\n' ${vcf} > pos.tsv
    echo "records: \$(wc -l < pos.tsv)"

    python3 - pos.tsv ${nchunk} <<'PY' > regions.txt
    import sys

    path = sys.argv[1]
    k    = int(sys.argv[2])

    n = sum(1 for _ in open(path))
    if n == 0:
        sys.exit("no records in " + path)

    # Record index at which each new chunk starts.
    targets = set(int(round(i * n / float(k))) for i in range(1, k))
    targets = set(t for t in targets if 0 < t < n)

    segments, counts = [], []
    cur, cnt = [], 0
    prev = None
    seg_chrom = seg_start = None
    pending = False

    with open(path) as fh:
        for i, line in enumerate(fh):
            c, p = line.split()
            p = int(p)
            if prev is None:
                seg_chrom, seg_start = c, p
            else:
                # Never cut between two records at the same site: multiallelic
                # rows share a POS and a POS-based range would claim both.
                if i in targets and (c, p) == prev:
                    pending = True
                cut = (i in targets or pending) and (c, p) != prev
                if c != prev[0] or cut:
                    cur.append((seg_chrom, seg_start, prev[1]))
                    seg_chrom, seg_start = c, p
                if cut:
                    segments.append(cur); counts.append(cnt)
                    cur, cnt = [], 0
                    pending = False
            prev = (c, p)
            cnt += 1

    cur.append((seg_chrom, seg_start, prev[1]))
    segments.append(cur); counts.append(cnt)

    letters = "abcdefghijklmnopqrstuvwxyz"
    for j, segs in enumerate(segments):
        reg = ",".join("%s:%d-%d" % s for s in segs)
        sys.stdout.write("%s\\t%s\\t%d\\n" % (letters[j], reg, counts[j]))
    PY

    cat regions.txt
    """

    stub:
    "printf 'a\\tchr20:1-1000\\t1\\n' > regions.txt"
}

process CONVERT_CHUNK {
    tag "${shard}/${subshard}${suffix}"
    label 'process_medium'

    publishDir params.outdir,
        mode: 'copy',
        overwrite: false,
        saveAs: { fn -> fn == 'versions.yml' ? null : "${shard}/${subshard}${suffix}/${fn}" }

    input:
    tuple val(shard), val(subshard), val(suffix), val(region), val(nrec), val(nchunk), path(vcf), path(idx)

    output:
    tuple val(shard), val("${subshard}${suffix}"), path("annotations.parquet"), emit: parquet
    path "columns.txt",     emit: columns
    path "schema_hash.tsv", emit: hash
    path "convert.log",     emit: log
    path "versions.yml",    emit: versions

    script:
    def af_arg = params.max_af ? "--max-af ${params.max_af}" : ''
    def ac_arg = params.max_ac ? "--max-ac ${params.max_ac}" : ''
    """
    export TMPDIR=\$PWD

    # --regions-overlap pos selects on POS alone. The default also returns
    # records that merely overlap the start of the range, so a deletion
    # spanning a boundary would be emitted in two chunks and double counted.
    # Probed rather than assumed: the flag is bcftools >=1.15.
    OVL=""
    if bcftools view 2>&1 | grep -q -- '--regions-overlap'; then
        OVL="--regions-overlap pos"
    fi

    set -o pipefail

    # Slice to a file, not a pipe: convert_annotations opens the input twice,
    # once for the CSQ header and once for the data. Doing it here also keeps
    # vcf2annotations.py in the container untouched.
    bcftools view \$OVL --regions '${region}' -Ob -o chunk.bcf ${vcf}
    bcftools index --csi chunk.bcf

    {
      echo "chunk:            ${subshard}${suffix}"
      echo "region:           ${region}"
      echo "records expected: ${nrec}"
      echo "records in slice: \$(bcftools index -n chunk.bcf 2>/dev/null || echo NA)"
    } > convert.log

    vcf2annotations.py convert_annotations \\
        chunk.bcf annotations.parquet \\
        --columns-file columns.txt \\
        --annotation-field ${params.annotation_field} \\
        ${af_arg} ${ac_arg} 2>&1 | tee -a convert.log

    # Peak disk is slice + intermediate TSV, so drop the slice as soon as the
    # conversion is done rather than leaving it for the work dir cleanup.
    rm -f chunk.bcf chunk.bcf.csi

    printf '%s\\t%s/%s%s\\t%s\\n' \\
        "\$(md5sum columns.txt | cut -d' ' -f1)" \\
        "${shard}" "${subshard}" "${suffix}" "\$(wc -l < columns.txt)" > schema_hash.tsv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bcftools: \$(bcftools --version | head -1 | sed 's/^bcftools //')
        polars: \$(python -c "import polars; print(polars.__version__)")
    END_VERSIONS
    """

    stub:
    "touch annotations.parquet columns.txt schema_hash.tsv convert.log versions.yml"
}

// ---------------- helpers ----------------

// resolveSibling stays inside the file's own filesystem provider. Building the
// path by string interpolation ("${vcf}.tbi") drops the s3:// scheme, and
// Nextflow then looks for a local file that does not exist.
// Returns null when neither index is present: without --region the VCFs are
// read sequentially and bcftools does not need one.
def indexFor(vcf) {
    for (ext in ['.tbi', '.csi']) {
        def idx = vcf.resolveSibling("${vcf.name}${ext}")
        if (idx.exists()) return idx
    }
    return null
}

// --chunked -> number of chunks per subshard, 0 meaning do not chunk.
// Accepts a boolean (Nextflow converts '--chunked true' for us) or a count,
// so the feature and its one tunable stay behind a single parameter.
def nChunks() {
    def v = params.chunked
    if (v == null || v == false) return 0
    if (v == true) return 4
    def s = v.toString().trim().toLowerCase()
    if (s in ['true', 'yes', 'on'])        return 4
    if (s in ['false', 'no', 'off', ''])   return 0
    if (s.isInteger() && s.toInteger() > 1) return s.toInteger()
    error "--chunked expects true/false or a chunk count greater than 1, got '${v}'"
}

// ---------------- workflow ----------------

workflow {
    DIAGNOSE()
    if (params.help) { helpMessage(); exit 0 }
    if (!params.annotation_root) { exit 1, "Missing --annotation_root" }

    def k = nChunks()

    ch_vcf = Channel
        .fromPath(
            "${params.annotation_root}/${params.shards}/${params.subshards}/${params.vcf_name}",
            checkIfExists: true
        )
        .map { vcf ->
            def sub_dir  = vcf.parent
            def subshard = sub_dir.name
            def shard    = sub_dir.parent.name
            // Index optional: an empty list stages nothing, so a missing or
            // stale index no longer kills the task. Except under --chunked,
            // where --regions is an index seek and there is nothing to seek in
            // without one.
            def idx = indexFor(vcf)
            if (k && !idx) error "--chunked needs an index: no .tbi/.csi beside ${vcf}"
            tuple(shard, subshard, vcf, idx ?: [])
        }

    if (k) {
        // One extra pass per subshard to find the cut points, then k tasks in
        // place of one. Same process graph from here on: the chunks are
        // ordinary subshards as far as publishing and CHECK_SCHEMA care.
        PARTITION_VCF(ch_vcf.map { shard, subshard, vcf, idx ->
            tuple(shard, subshard, vcf, idx, k)
        })

        ch_chunks = PARTITION_VCF.out
            .flatMap { shard, subshard, vcf, idx, regions ->
                regions.readLines()
                       .findAll { it?.trim() }
                       .collect { line ->
                           def f = line.split('\t')
                           // suffix, region, records-in-chunk
                           tuple(shard, subshard, f[0], f[1], f[2], k, vcf, idx)
                       }
            }

        CONVERT_CHUNK(ch_chunks)
        ch_hash    = CONVERT_CHUNK.out.hash
        ch_columns = CONVERT_CHUNK.out.columns
        ch_versions= CONVERT_CHUNK.out.versions
    }
    else {
        CONVERT_ANNOTATIONS(ch_vcf)
        ch_hash    = CONVERT_ANNOTATIONS.out.hash
        ch_columns = CONVERT_ANNOTATIONS.out.columns
        ch_versions= CONVERT_ANNOTATIONS.out.versions
    }

    // collectFile concatenates the one-line hashes into a single file without
    // staging every columns.txt into the checking task.
    //
    // Under --chunked the file is named for the run: chunks are normally a
    // rerun of one subshard, and writing to schema_hashes.tsv would overwrite
    // the full sweep's record of the other 3100 with four lines.
    ch_hashes = ch_hash.collectFile(
        name: k ? 'schema_hashes_chunked.tsv' : 'schema_hashes.tsv', sort: true,
        storeDir: "${params.outdir}/pipeline_info"
    )

    CHECK_SCHEMA(ch_hashes, ch_columns.first())

    ch_versions.first().collectFile(
        name: 'versions.yml', storeDir: "${params.outdir}/pipeline_info"
    )
}

workflow.onComplete {
    log.info """
    Completed : ${workflow.success ? 'OK' : 'FAILED'}
    Duration  : ${workflow.duration}
    Published : ${params.outdir}
    """.stripIndent()
}
