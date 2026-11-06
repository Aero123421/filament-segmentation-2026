"""Verify (or download) the bundled weights against ``weights/manifest.json``.

The same ``weights/`` layout serves both hosting methods:

- weights committed in git: run ``python fetch_weights.py`` to verify every
  file's SHA-256 and size;
- weights attached to a GitHub Release: download this repository without the
  weights, then run ``python fetch_weights.py --base-url <release-files-url>``
  to fetch each missing file from ``<base-url>/<path>`` and verify it.

Only the Python standard library is used.  Exit status is 0 when every file
in the manifest is present and intact, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_files(manifest: dict) -> list[dict]:
    """File entries of either manifest layout (``files`` or the default views)."""
    if "files" in manifest:
        return list(manifest["files"])
    entries = list(manifest.get("weights", [])) + list(manifest.get("refiner_weights", []))
    if isinstance(manifest.get("tree_model"), dict):
        entries.append(manifest["tree_model"])
    for model in (manifest.get("tree_models") or {}).values():
        if model not in entries:
            entries.append(model)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights-dir", type=Path, default=Path(__file__).resolve().parent / "weights",
                        help="weights directory holding manifest.json (default: next to this script)")
    parser.add_argument("--base-url", default="",
                        help="download missing files from <base-url>/<path> (GitHub Release files URL)")
    parser.add_argument("--config", default="",
                        help="only check the files used by this configuration (e.g. P3 or P)")
    args = parser.parse_args()

    manifest_path = args.weights_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = _manifest_files(manifest)
    if args.config:
        entries = [entry for entry in entries if args.config in (entry.get("used_by") or [manifest.get("default")])]
    failures = 0
    for entry in entries:
        path = args.weights_dir / entry["path"]
        ok = path.exists() and path.stat().st_size == entry["size_bytes"] and _sha256(path) == entry["sha256"]
        if not ok and args.base_url:
            url = f"{args.base_url.rstrip('/')}/{entry['path']}"
            print(f"downloading {entry['path']} from {url}")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:  # noqa: S310 (user-supplied URL)
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
            tmp.replace(path)
            ok = path.stat().st_size == entry["size_bytes"] and _sha256(path) == entry["sha256"]
        print(f"{'ok  ' if ok else 'FAIL'} {entry['path']} ({entry['size_bytes']} bytes)")
        failures += 0 if ok else 1
    total = sum(entry["size_bytes"] for entry in entries)
    if failures:
        print(f"{failures} of {len(entries)} weight files are missing or corrupt; see weights/LICENSE.md")
        return 1
    print(f"all {len(entries)} weight files verified ({total / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
