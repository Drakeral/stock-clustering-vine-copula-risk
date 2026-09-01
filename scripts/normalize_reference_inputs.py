#!/usr/bin/env python3
"""Remove volatile wrapper timestamps from existing reference inputs.

This one-time migration does not alter provider result rows. It makes the local
JSON wrapper match the deterministic format now emitted by the downloader and
refreshes the ignored operational manifest's corresponding hashes and sizes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from scripts.download_market_data import inspect_reference_file
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from download_market_data import inspect_reference_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normalize_reference_inputs(reference_root: Path, manifest_path: Path) -> tuple[int, int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_key = "reference_downloads" if "reference_downloads" in manifest else "reference_files"
    records = manifest.get(record_key, [])
    by_key = {(str(row["ticker"]), str(row["event_type"])): row for row in records}
    if len(by_key) != len(records):
        raise ValueError("Operational manifest has duplicate reference identities")

    files = sorted(reference_root.glob("*/*.json"))
    changed = 0
    seen: set[tuple[str, str]] = set()
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ticker = str(payload.get("ticker", ""))
        event_type = str(payload.get("event_type", ""))
        key = (ticker, event_type)
        if key not in by_key:
            raise ValueError(f"Reference file has no operational-manifest record: {path}")
        seen.add(key)
        if "downloaded_at_utc" in payload:
            del payload["downloaded_at_utc"]
            _write_json_atomic(path, payload)
            changed += 1
        metadata = inspect_reference_file(path, event_type, ticker)
        by_key[key].update(metadata)

    missing = sorted(set(by_key) - seen)
    if missing:
        raise ValueError(f"Operational manifest references absent files: {missing}")
    _write_json_atomic(manifest_path, manifest)
    return len(files), changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=PROJECT_ROOT / "data/raw/market/reference",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/raw/market/download_manifest.json",
    )
    args = parser.parse_args()
    total, changed = normalize_reference_inputs(args.reference_root, args.manifest)
    print(f"reference_files={total} normalized={changed}")


if __name__ == "__main__":
    main()
