#!/usr/bin/env nextflow
/*
 * AggV3 VEP-annotated msVCF -> long-format annotation parquet, one task per
 * subshard.
 *
 *   CONVERT_ANNOTATIONS  dragen.gel.annotated.vcf.gz -> annotations_<FIELD>.parquet
 *   CHECK_SCHEMA         every subshard's column list -> schema_<FIELD>.tsv
 *
 * One parquet per annotation field, each from its own split-vep pass. A single
 * pass carrying the other field as a raw string reprinted that whole block on
 * every consequence row -- ~40 copies per variant on this release.
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

// INFO fields to expand, one parquet each. Comma-separated. Each gets its own
// split-vep pass, so neither field's block is reprinted on the other's rows.
params.annotation_fields = 'CSQ,ANN'

// site filters, applied in bcftools during extraction
params.max_af          = null
params.max_ac          = null

params.help            = false

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
      --annotation_fields INFO fields to expand      [${params.annotation_fields}]
      --max_af            Drop sites above this AF   [${params.max_af}]
      --max_ac            Drop sites above this AC   [${params.max_ac}]
      --outdir            Publish directory          [${params.outdir}]

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
    tuple val(shard), val(subshard), path("annotations_*.parquet"), emit: parquet
    path "columns_*.txt",   emit: columns
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
        ${vcf} annotations \\
        --columns-prefix columns \\
        --fields ${params.annotation_fields} \\
        ${af_arg} ${ac_arg} 2>&1 | tee convert.log

    # One short line per subshard instead of shipping 3100 column lists to a
    # single task. On awsbatch every staged input is a real download, so the
    # hash keeps CHECK_SCHEMA a genuinely cheap task.
    # One line per field per subshard: field, md5, subshard, ncols. The field
    # is a grouping key in CHECK_SCHEMA -- CSQ and ANN have different column
    # lists by construction and must not be compared against each other.
    : > schema_hash.tsv
    for f in columns_*.txt; do
        field="\${f#columns_}"; field="\${field%.txt}"
        printf '%s\\t%s\\t%s/%s\\t%s\\n' \\
            "\$field" \\
            "\$(md5sum "\$f" | cut -d' ' -f1)" \\
            "${shard}" "${subshard}" "\$(wc -l < "\$f")" >> schema_hash.tsv
    done

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bcftools: \$(bcftools --version | head -1 | sed 's/^bcftools //')
        polars: \$(python -c "import polars; print(polars.__version__)")
    END_VERSIONS
    """

    stub:
    """
    for f in \$(echo ${params.annotation_fields} | tr ',' ' '); do
        touch annotations_\${f}.parquet columns_\${f}.txt
    done
    touch schema_hash.tsv convert.log versions.yml
    """
}

process CHECK_SCHEMA {
    label 'process_low'

    publishDir "${params.outdir}/pipeline_info", mode: 'copy', overwrite: true

    input:
    path hashes            // field, md5, shard/subshard, ncols -- one line per field per subshard
    path 'ref/*'           // one subshard's column list per field

    output:
    path "schema_*.tsv"
    path "schema_check.log"
    path "schema_variants.tsv", optional: true

    script:
    """
    #!/usr/bin/env python3
    import collections, glob, os, sys

    # Grouped by field first: CSQ and ANN have different column lists by
    # construction, and comparing them against each other would report a
    # mismatch on every run.
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    ncols  = collections.defaultdict(dict)
    for line in open("${hashes}"):
        if not line.strip():
            continue
        field, h, name, n = line.rstrip("\\n").split("\\t")
        groups[field][h].append(name)
        ncols[field][h] = n

    lines, bad = [], False
    for path in sorted(glob.glob("ref/columns_*.txt")):
        field = os.path.basename(path)[len("columns_"):-len(".txt")]
        ref = [c for c in open(path).read().splitlines() if c]
        with open(f"schema_{field}.tsv", "w") as out:
            for i, c in enumerate(ref):
                out.write(f"{i}\\t{c}\\n")
        g = groups.get(field, {})
        lines += [
            f"[{field}] subshards      : {sum(len(v) for v in g.values())}",
            f"[{field}] schema variants: {len(g)}",
            f"[{field}] columns        : {len(ref)}",
        ]
        if len(g) > 1:
            bad = True

    if bad:
        # Ordered by size: the majority schema first, then the outliers with a
        # named example each, so the next step is to diff two concrete files.
        with open("schema_variants.tsv", "w") as out:
            for field, g in sorted(groups.items()):
                if len(g) < 2:
                    continue
                for h, names in sorted(g.items(), key=lambda kv: -len(kv[1])):
                    out.write(f"{field}\\t{h}\\t{len(names)}\\t"
                              f"{ncols[field][h]}\\t{names[0]}\\n")
        lines += ["", "Subshards do not share a column schema within a field, "
                      "so those parquets will not concatenate. See "
                      "schema_variants.tsv for the group sizes and one example "
                      "subshard each."]

    open("schema_check.log", "w").write("\\n".join(lines) + "\\n")
    print("\\n".join(lines))
    if bad:
        sys.exit(1)
    """

    stub:
    "touch schema_CSQ.tsv schema_check.log"
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

// ---------------- workflow ----------------

workflow {
    DIAGNOSE()
    if (params.help) { helpMessage(); exit 0 }
    if (!params.annotation_root) { exit 1, "Missing --annotation_root" }

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
            // stale index no longer kills the task.
            tuple(shard, subshard, vcf, indexFor(vcf) ?: [])
        }

    CONVERT_ANNOTATIONS(ch_vcf)

    // collectFile concatenates the one-line hashes into a single file without
    // staging every columns.txt into the checking task.
    ch_hashes = CONVERT_ANNOTATIONS.out.hash.collectFile(
        name: 'schema_hashes.tsv', sort: true,
        storeDir: "${params.outdir}/pipeline_info"
    )

    // .first() emits one task's list of columns_<FIELD>.txt -- all of them,
    // one per field, staged into ref/ so the check covers every field.
    CHECK_SCHEMA(ch_hashes, CONVERT_ANNOTATIONS.out.columns.first())

    CONVERT_ANNOTATIONS.out.versions.first().collectFile(
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
