#!/usr/bin/env python3
"""Build the WebDataset shard manifest for the PhenoVision iNaturalist pull.

Reads the shared iNaturalist Open Data snapshot (gzipped TSVs) and emits:
  manifests/full_frame.parquet              one row per angiosperm photo, + shard and batch_j
  manifests/shard_lists/shard=N/data_0.csv  8-col TSV consumed by fetch_shard.py
  manifests/full_summary.txt                counts, sizes, license mix

Replaces the R targets download stage (filter_angio_taxa / _observations / _photos in
R/metadata_steps_download.R). Done in duckdb rather than R because Vulcan's R lacks
targets/arrow/duckdb/crew, and standing those up means building arrow from source.

Sharding is RANK-BASED on photo_id (SHARD_SIZE photos/shard, ordered by photo_id), which
is append-stable: a later snapshot's new (higher) photo_ids fall into NEW shards, leaving
existing shards byte-identical, so refreshing only ever ADDS shards.
"""
import argparse
import os
import sys
import textwrap

import duckdb

SHARD_SIZE = 10_000
BATCH_SIZE = 100_000      # legacy batch_j, kept so the pre-shard file_name layout still works
ANGIO_ROOT = "47125"      # iNat taxon id for angiosperms (Magnoliopsida)

# Photo licenses admitted into the shard store. NOT cosmetic.
#
# The pull creates ~100M re-encoded WebP copies packed into redistributable tar shards.
# A JPEG re-encoded to WebP q82 is plausibly an *adaptation*, and NoDerivatives (ND)
# forbids distributing adaptations; ShareAlike (SA) would additionally require any
# derivative to be SA-licensed, colliding with PhenoVision's MIT license.
#
# This whitelist also makes the corpus match what the published model cards already claim
# ("Images under CC-0, CC-BY, or CC-BY-NC licenses" -- phenovision_README.md:130), which
# the original pipeline never enforced: it had no license filter at all.
#
# PD is public domain -- the most permissive tag there is -- so it belongs here, not with
# the exclusions.
#
# Note iNaturalist's Open Data export is NOT limited to open licenses. Measured against the
# 2026-08-26 snapshot it carries 9.3M CC-BY-NC-ND, 9.2M CC-BY-NC-SA, 6.7M CC-BY-SA and
# 1.2M CC-BY-ND photos. Matching is case-insensitive: the export uses upper case, while
# iNat-leps lower-cased its copy.
DEFAULT_LICENSES = "CC0,CC-BY,CC-BY-NC,PD"


def reader(snap, name, cols):
    path = os.path.join(snap, name + ".csv.gz")
    if not os.path.exists(path):
        sys.exit("FATAL: missing " + path)
    collist = ", ".join("'%s': '%s'" % (c, t) for c, t in cols)
    return ("read_csv('%s', delim='\t', header=true, compression='gzip', "
            "columns={%s}, ignore_errors=true)" % (path, collist))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="dir holding taxa/observations/photos.csv.gz")
    ap.add_argument("--out", required=True, help="manifests/ output dir")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--memory", default="56GB")
    ap.add_argument("--temp", default=None, help="duckdb spill dir (default: <out>/.duckdb_tmp)")
    ap.add_argument("--licenses", default=DEFAULT_LICENSES,
                    help="comma-separated photo licenses to ADMIT, case-insensitive "
                         "(default: %s). Pass 'ALL' to disable filtering." % DEFAULT_LICENSES)
    a = ap.parse_args()

    snap = a.snapshot.rstrip("/")
    out = a.out.rstrip("/")
    sl = os.path.join(out, "shard_lists")
    os.makedirs(sl, exist_ok=True)
    tmp = a.temp or os.path.join(out, ".duckdb_tmp")
    os.makedirs(tmp, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=%d" % a.threads)
    con.execute("PRAGMA memory_limit='%s'" % a.memory)
    con.execute("PRAGMA temp_directory='%s'" % tmp)

    taxa = reader(snap, "taxa", [
        ("taxon_id", "BIGINT"), ("ancestry", "VARCHAR"), ("rank_level", "VARCHAR"),
        ("rank", "VARCHAR"), ("name", "VARCHAR"), ("active", "VARCHAR")])
    obs = reader(snap, "observations", [
        ("observation_uuid", "VARCHAR"), ("observer_id", "BIGINT"),
        ("latitude", "DOUBLE"), ("longitude", "DOUBLE"),
        ("positional_accuracy", "VARCHAR"), ("taxon_id", "BIGINT"),
        ("quality_grade", "VARCHAR"), ("observed_on", "VARCHAR"),
        ("anomaly_score", "VARCHAR")])
    pho = reader(snap, "photos", [
        ("photo_uuid", "VARCHAR"), ("photo_id", "BIGINT"),
        ("observation_uuid", "VARCHAR"), ("observer_id", "BIGINT"),
        ("extension", "VARCHAR"), ("license", "VARCHAR"),
        ("width", "VARCHAR"), ("height", "VARCHAR"), ("position", "VARCHAR")])

    # --- 1. angiosperm taxa ------------------------------------------------------
    # The original awk used  $2 ~ /47125/  -- a SUBSTRING regex, so an ancestry containing
    # e.g. 471250 or 147125 matched too. We use a proper path-component test and REPORT the
    # difference rather than silently changing the corpus out from under the published model.
    print("[1/5] angiosperm taxa ...", flush=True)
    con.execute("""
      CREATE TEMP TABLE angio_taxa AS
      SELECT taxon_id,
             ('/' || ancestry || '/') LIKE '%/{root}/%' AS strict_hit,
             ancestry LIKE '%{root}%'                   AS loose_hit
      FROM {taxa}
      WHERE rank IN ('species','subspecies','variety')
        AND ancestry LIKE '%{root}%'
    """.format(root=ANGIO_ROOT, taxa=taxa))
    strict, loose = con.execute(
        "SELECT count(*) FILTER (WHERE strict_hit), count(*) FILTER (WHERE loose_hit) "
        "FROM angio_taxa").fetchone()
    print("      strict (path-component) : {:,}".format(strict))
    print("      loose  (old awk substr) : {:,}".format(loose))
    print("      DIFFERENCE              : {:,} taxa the old filter over-included".format(loose - strict))
    con.execute("DELETE FROM angio_taxa WHERE NOT strict_hit")

    # --- 2. research-grade observations of those taxa ----------------------------
    print("[2/5] research-grade observations ...", flush=True)
    con.execute("""
      CREATE TEMP TABLE angio_obs AS
      SELECT o.observation_uuid, o.latitude, o.longitude, o.positional_accuracy,
             o.taxon_id, o.observed_on
      FROM {obs} o
      SEMI JOIN angio_taxa t ON o.taxon_id = t.taxon_id
      WHERE o.quality_grade = 'research'
    """.format(obs=obs))
    print("      observations: {:,}".format(
        con.execute("SELECT count(*) FROM angio_obs").fetchone()[0]))

    # --- 3. photos of those observations, license-filtered, sharded ---------------
    print("[3/5] photos + license filter + shard assignment ...", flush=True)
    if a.licenses.strip().upper() == "ALL":
        lic_pred = "TRUE"
        print("      licenses: ALL (no filter) -- see DEFAULT_LICENSES for why this is risky")
    else:
        admitted = [x.strip().upper() for x in a.licenses.split(",") if x.strip()]
        lic_pred = "upper(p.license) IN (%s)" % ", ".join("'%s'" % x for x in admitted)
        print("      licenses admitted: %s" % ", ".join(admitted))

    # Count what the filter removes BEFORE applying it, so the exclusion is always
    # recorded rather than silently disappearing from the corpus.
    excluded = con.execute("""
      SELECT p.license, count(*) c
      FROM {pho} p SEMI JOIN angio_obs o ON p.observation_uuid = o.observation_uuid
      WHERE NOT ({lic})
      GROUP BY 1 ORDER BY c DESC
    """.format(pho=pho, lic=lic_pred)).fetchall()
    n_excluded = sum(c for _, c in excluded)
    if excluded:
        print("      EXCLUDED by license: {:,} photos".format(n_excluded))
        for k, c in excluded:
            print("        {:<14} {:>12,}".format(k if k else "<empty>", c))

    con.execute("""
      CREATE TEMP TABLE frame AS
      SELECT p.photo_id, p.photo_uuid, p.observation_uuid, p.observer_id,
             p.extension, p.license, p.width, p.height, p.position,
             o.taxon_id, o.latitude, o.longitude, o.positional_accuracy, o.observed_on,
             CAST(FLOOR((ROW_NUMBER() OVER (ORDER BY p.photo_id) - 1) / {shard}) AS INTEGER) AS shard,
             CAST(FLOOR((ROW_NUMBER() OVER (ORDER BY p.photo_id) - 1) / {batch}) AS INTEGER) + 1 AS batch_j
      FROM {pho} p
      JOIN angio_obs o USING (observation_uuid)
      WHERE {lic}
    """.format(shard=SHARD_SIZE, batch=BATCH_SIZE, pho=pho, lic=lic_pred))
    n_photos, n_shards, n_taxa, n_obs = con.execute(
        "SELECT count(*), max(shard)+1, count(DISTINCT taxon_id), "
        "count(DISTINCT observation_uuid) FROM frame").fetchone()
    print("      photos {:,} | shards {:,} | taxa {:,} | observations {:,}".format(
        n_photos, n_shards, n_taxa, n_obs))

    # FLOOR, not CAST(x/N AS INTEGER): duckdb's CAST ROUNDS, which silently made iNat-leps'
    # shard 0 half-size. Assert the invariant rather than trusting it.
    s0 = con.execute("SELECT count(*) FROM frame WHERE shard = 0").fetchone()[0]
    assert s0 == SHARD_SIZE or n_shards == 1, \
        "shard 0 has %d, expected %d (rounding bug?)" % (s0, SHARD_SIZE)
    print("      shard 0 size {:,}  (FLOOR bucketing verified)".format(s0))

    # --- 4. outputs ---------------------------------------------------------------
    print("[4/5] writing parquet + shard lists ...", flush=True)
    fp = os.path.join(out, "full_frame.parquet")
    con.execute("COPY frame TO '%s' (FORMAT parquet, COMPRESSION zstd)" % fp)

    # Columns in the exact order fetch_shard.py unpacks them (it reads the first 7).
    con.execute("""
      COPY (SELECT photo_id, extension, observation_uuid, taxon_id, license,
                   latitude, longitude, shard
            FROM frame ORDER BY shard, photo_id)
      TO '{sl}' (FORMAT csv, DELIMITER '\t', HEADER false,
                 PARTITION_BY (shard), OVERWRITE_OR_IGNORE, FILENAME_PATTERN 'data_{{i}}')
    """.format(sl=sl))

    # --- 5. summary ----------------------------------------------------------------
    print("[5/5] summary ...", flush=True)
    lic = con.execute(
        "SELECT license, count(*) c FROM frame GROUP BY 1 ORDER BY c DESC").fetchall()
    est_gb = n_photos * 45.0 / 1024 / 1024   # 45 KB/img measured, notes/image_compression_report.md
    summary = textwrap.dedent("""\
        full frame: {n_photos:,} photos | {n_shards:,} shards (x{shard_size}) | {n_taxa:,} taxa | {n_obs:,} observations
        est. sharded size @ ~45 KB/img (webp q82, measured): {est_gb:,.0f} GB
        angiosperm taxa: {strict:,} strict / {loose:,} loose (old awk over-included {over:,})
        snapshot: {snap}
        licenses admitted: {adm}
        license mix: {lic}
        EXCLUDED by license: {n_excl:,} photos{excl_detail}
        """).format(
        n_photos=n_photos, n_shards=n_shards, shard_size=SHARD_SIZE, n_taxa=n_taxa,
        n_obs=n_obs, est_gb=est_gb, strict=strict, loose=loose, over=loose - strict,
        snap=snap, adm=a.licenses,
        lic=", ".join("%s=%s" % (k, format(v, ",")) for k, v in lic),
        n_excl=n_excluded,
        excl_detail=("" if not excluded else " ("
                     + ", ".join("%s=%s" % (k, format(c, ",")) for k, c in excluded) + ")"))
    with open(os.path.join(out, "full_summary.txt"), "w") as fh:
        fh.write(summary)
    print(summary)
    print("wrote " + fp)
    print("wrote " + sl + "/shard=*/data_0.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
