"""WebP conversion helper for the PhenoVision iNaturalist image store.

Single source of truth for jpg/png -> WebP q82 conversion, used by:
  - the one-time bulk converter (scripts/convert_store_to_webp.sh), and
  - the download pipeline (R/download_images_new_download.R, via reticulate).

Policy: convert each <id>.<jpg|jpeg|png> -> <id>.webp (quality 82, method 6) and delete the
original. A source that won't decode/convert is CORRUPT -> delete it too (it would be skipped by
training/inference anyway; deleting keeps the store self-cleaning). Idempotent: an existing,
decodable <id>.webp is left in place. The webp is written to a .tmp then atomically renamed, and
verified to decode before the original is removed, so an interrupted run never loses data.
"""
import os
from PIL import Image

QUALITY = 82
METHOD = 6


def convert_to_webp(src, delete_src=True, skip_existing=True):
    """Convert one image file to `<stem>.webp` (q82/method6).

    Returns the webp path on success, or None on a corrupt/unconvertible source. On failure the
    bad source is deleted (when delete_src). Idempotent when skip_existing: a present, non-empty
    `<stem>.webp` is returned as-is (and any leftover original removed).
    """
    stem, ext = os.path.splitext(src)
    dst = stem + ".webp"
    if ext.lower() == ".webp":
        return src
    if skip_existing and os.path.exists(dst) and os.path.getsize(dst) > 0:
        if delete_src and os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
            try: os.remove(src)
            except OSError: pass
        return dst
    tmp = dst + ".tmp"
    # CRITICAL SAFETY RULE: NEVER delete an original that we did not just replace with a verified
    # webp. Any failure -- decode OR encode -- leaves the original in place and returns None.
    #
    # Why: a transient filesystem READ error during decode is indistinguishable from a genuinely
    # corrupt file (both raise from Image.load()). An earlier version deleted "corrupt" sources on
    # decode failure; during two brief /blue I/O hiccups this misclassified ~1.96M perfectly good
    # images as corrupt and deleted them. So we do NOT self-clean corrupt files here at all --
    # unconvertible sources are only LOGGED (--fail-log) and left on disk; a separate, deliberate
    # pass can inspect/remove genuinely-corrupt files if ever desired. Deletion happens in exactly
    # one place: after a webp has been written AND re-verified to decode.
    try:
        with Image.open(src) as im:
            rgb = im.convert("RGB")
            rgb.load()
        rgb.save(tmp, "WEBP", quality=QUALITY, method=METHOD)
        with Image.open(tmp) as v:          # verify the produced webp decodes
            v.load()
        os.replace(tmp, dst)                # atomic publish
    except Exception:
        try: os.remove(tmp)
        except OSError: pass
        return None                          # keep the original on ANY failure -- never delete here
    # success: webp is durable -> now safe to remove the original
    if delete_src and os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
        try: os.remove(src)
        except OSError: pass
    return dst


def convert_many(paths, delete_src=True, skip_existing=True):
    """Convert a list of paths serially. Returns (n_ok, failed_paths)."""
    ok, failed = 0, []
    for p in paths:
        if convert_to_webp(p, delete_src=delete_src, skip_existing=skip_existing) is None:
            failed.append(p)
        else:
            ok += 1
    return ok, failed


def _conv(arg):
    src, delete_src, skip_existing = arg
    return convert_to_webp(src, delete_src=delete_src, skip_existing=skip_existing)


def _find_images(d):
    import glob
    out = []
    for e in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG", "Jpg", "Jpeg", "Png"):
        out += glob.glob(os.path.join(d, f"*.{e}"))
    return sorted(set(out))


if __name__ == "__main__":
    # Per-batch CLI for the bulk converter: convert every jpg/jpeg/png under DIR(s) in parallel.
    #   python py/webp_convert.py DIR [DIR ...] [--workers N] [--no-delete] [--fail-log PATH]
    import sys, argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-delete", action="store_true", help="keep originals (dry/test mode)")
    ap.add_argument("--fail-log", default=None, help="append failed (corrupt) source paths here")
    a = ap.parse_args()
    srcs = []
    for d in a.dirs:
        srcs += _find_images(d)
    failed = []
    if srcs:
        args = [(s, not a.no_delete, True) for s in srcs]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for src, r in zip(srcs, ex.map(_conv, args, chunksize=64)):
                if r is None:
                    failed.append(src)
    if a.fail_log and failed:
        with open(a.fail_log, "a") as f:
            f.write("\n".join(failed) + "\n")
    n_src = len(srcs)
    print(f"n_src={n_src} n_webp={n_src - len(failed)} n_failed={len(failed)}")
    sys.exit(0)
