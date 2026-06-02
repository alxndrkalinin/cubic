"""Create a Zenodo deposition and upload cubic's deconvolution example data.

Usage
-----
1. Get a personal access token (with ``deposit:write`` and
   ``deposit:actions`` scopes) from https://zenodo.org/account/settings/applications/
   (or https://sandbox.zenodo.org/account/settings/applications/ for testing).
2. Export it: ``export ZENODO_TOKEN=<token>``
3. Run from the repo root:

   .. code-block:: bash

       # sandbox dry-run (recommended first):
       python scripts/zenodo_upload.py --sandbox

       # real upload:
       python scripts/zenodo_upload.py

       # by default, the deposition is left as a DRAFT for the user to
       # inspect and publish from the Zenodo UI. Pass --publish to
       # publish immediately (irreversible; metadata becomes locked
       # except via a versioned update).
       python scripts/zenodo_upload.py --publish

The script:
- POSTs a new deposition (empty),
- attaches metadata loaded from ``scripts/zenodo_metadata.json``,
- streams each file in ``FILES_TO_UPLOAD`` to the deposition's bucket,
- verifies returned checksums against locally computed sha256/md5,
- prints the draft link (and final DOI if ``--publish`` is set).
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "zenodo_upload.py requires the `requests` package, which is not a "
        "cubic runtime dependency. Install it into your environment first: "
        "`pip install requests` (or `uv pip install requests`)."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = REPO_ROOT / "scripts" / "zenodo_metadata.json"

FILES_TO_UPLOAD: list[tuple[Path, str]] = [
    (REPO_ROOT / "examples/data/astr_vpa_hoechst.tif", "astr_vpa_hoechst.tif"),
    (
        REPO_ROOT / "examples/data/astr_vpa_hoechst_psf_na095_cropped.tif",
        "astr_vpa_hoechst_psf_na095_cropped.tif",
    ),
]

EXPECTED_SHA256: dict[str, str] = {
    "astr_vpa_hoechst.tif": "234533100739f31ea31b78c380bae6cc2ea6b9cebec2c3160eedb89c36967cdc",
    "astr_vpa_hoechst_psf_na095_cropped.tif": "7a5bfef942a52b8eb683286992058e647937276cd6e5fd43bc32ef6e3134feed",
}


def hash_file(path: Path, algos: tuple[str, ...] = ("sha256", "md5")) -> dict[str, str]:
    """Stream ``path`` and return one hex digest per algorithm in ``algos``."""
    hashers = {a: hashlib.new(a) for a in algos}
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            for h in hashers.values():
                h.update(chunk)
    return {a: hashers[a].hexdigest() for a in algos}


def main() -> int:
    """Upload ``FILES_TO_UPLOAD`` to a new Zenodo deposition with the configured metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use sandbox.zenodo.org instead of zenodo.org.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the deposition after upload (irreversible).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=METADATA_PATH,
        help=f"Path to deposition metadata JSON (default: {METADATA_PATH}).",
    )
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ERROR: ZENODO_TOKEN env var is not set.", file=sys.stderr)
        return 2

    base = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"
    api = f"{base}/api"
    params = {"access_token": token}

    metadata = json.loads(args.metadata.read_text())
    print(f"Loaded metadata from {args.metadata}")
    print(f"  title: {metadata['metadata']['title']}")
    print(f"  creators: {len(metadata['metadata']['creators'])}")
    print(f"  license: {metadata['metadata']['license']}")

    for p, _ in FILES_TO_UPLOAD:
        if not p.is_file():
            print(f"ERROR: missing file {p}", file=sys.stderr)
            return 2

    print(f"\nUsing Zenodo endpoint: {base}")
    print("Creating empty deposition...")
    r = requests.post(f"{api}/deposit/depositions", params=params, json={}, timeout=30)
    r.raise_for_status()
    deposition = r.json()
    dep_id = deposition["id"]
    bucket_url = deposition["links"]["bucket"]
    html_url = deposition["links"]["html"]
    print(f"  deposition id: {dep_id}")
    print(f"  draft URL:     {html_url}")
    print(f"  bucket URL:    {bucket_url}")

    print("\nAttaching metadata...")
    r = requests.put(
        f"{api}/deposit/depositions/{dep_id}",
        params=params,
        json=metadata,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    print("  ok.")

    for path, remote_name in FILES_TO_UPLOAD:
        size_mb = path.stat().st_size / 1e6
        print(f"\nUploading {remote_name} ({size_mb:.1f} MB)...")
        hashes = hash_file(path)
        if (
            remote_name in EXPECTED_SHA256
            and hashes["sha256"] != EXPECTED_SHA256[remote_name]
        ):
            print(
                f"ERROR: local sha256 for {remote_name} does not match the "
                f"committed expected value. Local={hashes['sha256']}, "
                f"expected={EXPECTED_SHA256[remote_name]}.",
                file=sys.stderr,
            )
            return 3
        with path.open("rb") as fh:
            # (connect, read) — a 30-minute read timeout tolerates slow
            # uploads of large depositions (Zenodo's bucket API streams
            # chunked) but still surfaces stalled connections instead of
            # hanging the script indefinitely.
            r = requests.put(
                f"{bucket_url}/{remote_name}",
                params=params,
                data=fh,
                timeout=(10, 1800),
            )
        r.raise_for_status()
        info = r.json()
        zenodo_md5 = info.get("checksum", "").removeprefix("md5:")
        if zenodo_md5 != hashes["md5"]:
            print(
                f"ERROR: md5 mismatch for {remote_name}. "
                f"local={hashes['md5']} zenodo={zenodo_md5}.",
                file=sys.stderr,
            )
            return 4
        print(f"  ok. sha256={hashes['sha256']} md5={hashes['md5']}")

    if args.publish:
        print("\nPublishing deposition (irreversible)...")
        r = requests.post(
            f"{api}/deposit/depositions/{dep_id}/actions/publish",
            params=params,
            timeout=60,
        )
        r.raise_for_status()
        published = r.json()
        doi = published.get("doi") or published.get("metadata", {}).get("doi")
        concept_doi = published.get("conceptdoi")
        print("\nPublished.")
        print(f"  version DOI: {doi}")
        print(f"  concept DOI: {concept_doi}")
        print(f"  record URL:  {published['links'].get('record_html')}")
    else:
        print("\nDraft created. Inspect at:")
        print(f"  {html_url}")
        print("Rerun with --publish to publish, or click 'Publish' in the Zenodo UI.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
