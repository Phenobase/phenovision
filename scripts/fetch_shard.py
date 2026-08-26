#!/usr/bin/env python3
"""Fetch ONE WebDataset shard: download each photo from the iNaturalist Open Data S3
bucket, convert to WebP, and pack into a .tar.

WebDataset layout -- <photo_id>.webp and <photo_id>.json share a key, so together they
are one sample.

Ported from iNat-leps/scripts/fetch_shard.py, which pulled 25.5M images in ~5.7h at
~1,250 img/s through the Vulcan Squid proxy. Differences here:

  - WebP q82 / method 6, matching phenovision's existing store (py/webp_convert.py), NOT
    iNat-leps' q85. See notes/image_compression_report.md for why q82.
  - Retries with backoff. iNat-leps had NONE (a bare except, record the class, move on) and
    got away with it at a 0.0043% failure rate; this corpus is ~4x larger.
  - A richer JSON sidecar, INCLUDING `license`. That field is what makes it possible to
    exclude ShareAlike/NoDerivatives images at read time without rebuilding any tar.
    iNat-leps' journal records the sidecar decision saving a 29 GB metadata re-download;
    it is load-bearing here for a licensing reason too.

Nothing is ever written to disk as a loose file: bytes go S3 -> memory -> WebP -> tar.
Besides being fast, that is why the delete-on-failure bug that destroyed ~1.96M images in
the 2026-07-01 incident cannot recur here -- there is no original on disk to delete.

Usage: fetch_shard.py <shard_id>          (env: THREADS, QUALITY, RETRIES)
"""
import io
import json
import glob
import os
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

BASE = os.environ.get("PHENOVISION_STORE",
                      os.path.expanduser("~/projects/aip-dinnage/phenovision"))
SL = os.environ.get("SHARD_LISTS",
                    os.path.join(os.path.expanduser("~/phenovision"), "manifests", "shard_lists"))
OUT = os.environ.get("SHARD_OUT",
                     os.path.join(BASE, "data", "phenobase_inat_data", "images", "shards"))
S3 = "https://inaturalist-open-data.s3.amazonaws.com"

QUALITY = int(os.environ.get("QUALITY", "82"))   # matches py/webp_convert.py
METHOD = 6
TIMEOUT = int(os.environ.get("TIMEOUT", "30"))
THREADS = int(os.environ.get("THREADS", "16"))
RETRIES = int(os.environ.get("RETRIES", "3"))

os.makedirs(OUT, exist_ok=True)

shard = int(sys.argv[1])
tar_path = os.path.join(OUT, "shard-%05d.tar" % shard)
done_path = tar_path + ".done"
err_path = tar_path + ".err"

if os.path.exists(done_path):
    print("shard %d already done" % shard)
    sys.exit(0)

# The manifest writer (duckdb PARTITION_BY) may emit data_0.csv AND data_1.csv for a
# partition -- 541 of 9,805 shards did. Glob, do not assume a single file.
rows = []
for f in sorted(glob.glob(os.path.join(SL, "shard=%d" % shard, "*.csv"))):
    with open(f) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 7:
                rows.append(p[:7])

if not rows:
    print("shard %d: no rows" % shard)
    with open(done_path, "w") as fh:
        fh.write("ok=0 fail=0 (empty)\n")
    sys.exit(0)

sess = requests.Session()


def fetch_one(r):
    pid, ext, obs, taxon, lic, lat, lon = r
    url = "%s/photos/%s/medium.%s" % (S3, pid, ext)
    last = None
    for attempt in range(RETRIES):
        try:
            resp = sess.get(url, timeout=TIMEOUT)
            if resp.status_code == 404:
                # Withdrawn or deleted since the snapshot. Not retryable, and EXPECTED:
                # a re-download can never fully reproduce an older corpus.
                return pid, None, None, "HTTP404"
            resp.raise_for_status()
            im = Image.open(io.BytesIO(resp.content)).convert("RGB")
            buf = io.BytesIO()
            im.save(buf, "WEBP", quality=QUALITY, method=METHOD)
            meta = json.dumps({
                "photo_id": int(pid),
                "observation_uuid": obs,
                "taxon_id": int(taxon) if taxon else None,
                "license": lic,
                "extension": ext,          # original S3 format, for provenance
                "lat": float(lat) if lat else None,
                "lon": float(lon) if lon else None,
            }).encode()
            return pid, buf.getvalue(), meta, None
        except Exception as e:              # noqa: BLE001 - any failure is retryable once
            last = type(e).__name__
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt)    # 1s, 2s, 4s
    return pid, None, None, last


n_ok = n_fail = 0
errs = []
tmp = tar_path + ".tmp"
t0 = time.time()

# Downloads/converts run in a thread pool; the tar is written sequentially by this thread.
with ThreadPoolExecutor(max_workers=THREADS) as ex, tarfile.open(tmp, "w") as tar:
    for pid, web, meta, err in ex.map(fetch_one, rows):
        if err:
            n_fail += 1
            errs.append("%s\t%s" % (pid, err))
            continue
        ti = tarfile.TarInfo("%s.webp" % pid)
        ti.size = len(web)
        tar.addfile(ti, io.BytesIO(web))
        tj = tarfile.TarInfo("%s.json" % pid)
        tj.size = len(meta)
        tar.addfile(tj, io.BytesIO(meta))
        n_ok += 1

os.replace(tmp, tar_path)                   # atomic publish
if errs:
    with open(err_path, "w") as fh:
        fh.write("\n".join(errs) + "\n")
with open(done_path, "w") as fh:
    fh.write("ok=%d fail=%d\n" % (n_ok, n_fail))

dt = time.time() - t0
print("shard %d: ok=%d fail=%d in %.1fs (%.0f img/s) -> %s"
      % (shard, n_ok, n_fail, dt, n_ok / dt if dt else 0, os.path.basename(tar_path)))
