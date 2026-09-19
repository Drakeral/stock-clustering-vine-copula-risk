#!/usr/bin/env python3
"""Download immutable Massive daily aggregates and reference corporate actions.

Credentials are read only from environment variables. The script never writes
credentials or authenticated URLs to disk. Existing provider files are not
overwritten unless --overwrite is supplied.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9-3.10
    import tomli as tomllib

try:
    from scripts.pipeline_io import temporary_sibling, write_json_atomic
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from pipeline_io import temporary_sibling, write_json_atomic


KEY_DATE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})\.csv\.gz$")
EXPECTED_HEADER = {
    "ticker",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "window_start",
    "transactions",
}
MASSIVE_REST_HOSTS = frozenset({"api.massive.com"})
MASSIVE_S3_HOSTS = frozenset({"files.massive.com"})


def validated_https_url(
    url: str,
    *,
    allowed_hosts: frozenset[str],
    provider: str,
) -> urllib.parse.SplitResult:
    """Return a parsed provider URL after enforcing a narrow HTTPS boundary."""

    if not isinstance(url, str) or not url:
        raise ValueError(f"Invalid {provider} URL")
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid {provider} URL") from exc
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        raise ValueError(f"Untrusted {provider} URL origin")
    return parsed


def validated_s3_endpoint(url: str) -> str:
    """Return the canonical Massive S3 endpoint before credentials are attached."""

    parsed = validated_https_url(
        url,
        allowed_hosts=MASSIVE_S3_HOSTS,
        provider="Massive S3",
    )
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Untrusted Massive S3 endpoint")
    return f"https://{parsed.hostname}"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def project_path(project_root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else project_root / path


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def make_s3_client(config: dict[str, Any]):
    massive = config["massive"]
    endpoint = validated_s3_endpoint(massive["flat_file_endpoint"])
    access_key = require_env(massive["s3_access_key_env"])
    secret_key = require_env(massive["s3_secret_key_env"])
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 8, "mode": "adaptive"},
            connect_timeout=20,
            read_timeout=120,
            max_pool_connections=16,
        ),
    )


def iter_objects(
    client: Any,
    bucket: str,
    prefix: str,
    start: dt.date,
    end: dt.date,
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_objects_v2")
    for year in range(start.year, end.year + 1):
        year_prefix = f"{prefix.rstrip('/')}/{year}/"
        for page in paginator.paginate(Bucket=bucket, Prefix=year_prefix):
            for item in page.get("Contents", []):
                match = KEY_DATE.search(item["Key"])
                if not match:
                    continue
                file_date = dt.date.fromisoformat(match.group("date"))
                if start <= file_date <= end:
                    objects.append(
                        {
                            "date": file_date.isoformat(),
                            "key": item["Key"],
                            "size": int(item["Size"]),
                            "etag": str(item.get("ETag", "")).strip('"'),
                            "last_modified": item["LastModified"].isoformat(),
                        }
                    )
    objects.sort(key=lambda row: row["date"])
    if len({row["date"] for row in objects}) != len(objects):
        raise RuntimeError("Provider listing contains duplicate daily files")
    return objects


def inspect_daily_file(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().strip().split(",")
        row_count = sum(1 for line in handle if line.strip())
    if set(header) != EXPECTED_HEADER:
        raise RuntimeError(f"Unexpected schema in {path}: {header}")
    if row_count <= 0:
        raise RuntimeError(f"Daily aggregate is empty: {path}")
    return {"columns": sorted(header), "row_count": row_count}


def validate_daily_file(path: Path) -> None:
    """Backwards-compatible validation entry point."""

    inspect_daily_file(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path_for(raw_root: Path, key: str) -> Path:
    if not isinstance(key, str) or not key:
        raise ValueError("Provider object key must be a non-empty string")
    try:
        relative = Path(key).relative_to("us_stocks_sip/day_aggs_v1")
    except ValueError as exc:
        raise ValueError("Provider object key is outside the daily-aggregate prefix") from exc
    if relative == Path(".") or ".." in relative.parts:
        raise ValueError("Provider object key contains an unsafe relative component")
    destination_root = (raw_root / "day_aggs").resolve()
    destination = (destination_root / relative).resolve()
    try:
        destination.relative_to(destination_root)
    except ValueError as exc:
        raise ValueError("Provider object key resolves outside the raw-data directory") from exc
    return destination


def download_one(
    client: Any,
    bucket: str,
    project_root: Path,
    raw_root: Path,
    item: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    destination = local_path_for(raw_root, item["key"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    status = "existing"
    if overwrite or not destination.exists() or destination.stat().st_size != item["size"]:
        with temporary_sibling(destination) as temporary:
            client.download_file(bucket, item["key"], str(temporary))
            file_metadata = inspect_daily_file(temporary)
            temporary.replace(destination)
        status = "downloaded"
    else:
        file_metadata = inspect_daily_file(destination)
    return {
        **item,
        "local_path": str(destination.relative_to(project_root)),
        **file_metadata,
        "status": status,
        "sha256": sha256_file(destination),
    }


def unique_provider_tickers(universe_path: Path, security_master_path: Path) -> list[str]:
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    master = json.loads(security_master_path.read_text(encoding="utf-8"))
    tickers = {row["ticker"] for row in universe["constituents"]}
    for segments in master["mappings"].values():
        tickers.update(segment["ticker"] for segment in segments)
    return sorted(tickers)


def authenticated_json(url: str, api_key: str, attempts: int = 7) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    parsed = validated_https_url(
        url,
        allowed_hosts=MASSIVE_REST_HOSTS,
        provider="Massive REST",
    )
    query = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() != "apikey"
    ]
    query.append(("apiKey", api_key))
    authenticated_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), parsed.fragment)
    )
    safe_endpoint = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    # The URL origin is allowlisted before the credential is attached.
    request = urllib.request.Request(  # noqa: S310
        authenticated_url,
        headers={"Accept": "application/json", "User-Agent": "fe5110-research-pipeline/0.1"},
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise RuntimeError(
                    f"REST request failed with HTTP {exc.code} at {safe_endpoint}"
                ) from None
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, OSError):
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"REST request failed after {attempts} attempts at {safe_endpoint}"
                ) from None
        time.sleep(min(30.0, 1.5 * (2**attempt)))
    raise RuntimeError("Unreachable retry state")


def inspect_reference_file(path: Path, event_type: str, ticker: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported reference schema in {path}")
    if payload.get("event_type") != event_type or payload.get("ticker") != ticker:
        raise RuntimeError(f"Reference identity differs from path in {path}")
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"Reference results must be a list in {path}")
    required_date = "execution_date" if event_type == "splits" else "ex_dividend_date"
    missing_date_rows = sum(
        not isinstance(row, dict) or not row.get(required_date) for row in results
    )
    if missing_date_rows:
        raise RuntimeError(f"{path} has {missing_date_rows} rows without {required_date}")
    return {
        "rows": len(results),
        "payload_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "schema_version": payload["schema_version"],
        "sample_start": payload.get("sample_start"),
        "sample_end": payload.get("sample_end"),
    }


def download_reference_type(
    rest_base_url: str,
    api_key: str,
    project_root: Path,
    reference_dir: Path,
    event_type: str,
    ticker: str,
    start: str,
    end: str,
    overwrite: bool,
) -> dict[str, Any]:
    destination = reference_dir / event_type / f"{ticker.replace('.', '_')}.json"
    date_field = "execution_date" if event_type == "splits" else "ex_dividend_date"
    request_metadata = {
        "endpoint": f"{rest_base_url.rstrip('/')}/v3/reference/{event_type}",
        "parameters": {
            "ticker": ticker,
            f"{date_field}.gte": start,
            f"{date_field}.lte": end,
            "sort": date_field,
            "order": "asc",
            "limit": 1000,
        },
    }
    if destination.exists() and not overwrite:
        metadata = inspect_reference_file(destination, event_type, ticker)
        return {
            "ticker": ticker,
            "event_type": event_type,
            "local_path": str(destination.relative_to(project_root)),
            "request": request_metadata,
            **metadata,
            "status": "existing",
        }

    query = urllib.parse.urlencode(request_metadata["parameters"])
    next_url: str | None = f"{rest_base_url.rstrip('/')}/v3/reference/{event_type}?{query}"
    results: list[dict[str, Any]] = []
    while next_url:
        page = authenticated_json(next_url, api_key)
        page_results = page.get("results", [])
        if not isinstance(page_results, list) or any(
            not isinstance(row, dict) for row in page_results
        ):
            raise RuntimeError("Massive reference response has invalid results")
        results.extend(page_results)
        provider_next_url = page.get("next_url")
        if provider_next_url is not None and not isinstance(provider_next_url, str):
            raise RuntimeError("Massive reference response has invalid next_url")
        next_url = provider_next_url

    payload = {
        "schema_version": 1,
        "provider": "Massive",
        "event_type": event_type,
        "ticker": ticker,
        "sample_start": start,
        "sample_end": end,
        "results": results,
    }
    write_json_atomic(destination, payload)
    return {
        "ticker": ticker,
        "event_type": event_type,
        "local_path": str(destination.relative_to(project_root)),
        "request": request_metadata,
        **inspect_reference_file(destination, event_type, ticker),
        "status": "downloaded",
    }


def download_references(
    config: dict[str, Any],
    project_root: Path,
    start: str,
    end: str,
    overwrite: bool,
) -> list[dict[str, Any]]:
    massive = config["massive"]
    api_key = require_env(massive["rest_api_key_env"])
    paths = config["paths"]
    tickers = unique_provider_tickers(
        project_path(project_root, paths["universe_json"]),
        project_path(project_root, paths["security_master_json"]),
    )
    reference_dir = project_path(project_root, paths["corporate_actions_directory"])
    records = []
    total = len(tickers) * 2
    completed = 0
    for ticker in tickers:
        for event_type in ("splits", "dividends"):
            records.append(
                download_reference_type(
                    massive["rest_base_url"],
                    api_key,
                    project_root,
                    reference_dir,
                    event_type,
                    ticker,
                    start,
                    end,
                    overwrite,
                )
            )
            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f"Reference downloads: {completed}/{total}", flush=True)
    return records


def public_input_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Remove operational timestamps/statuses while preserving input identity."""

    daily_fields = (
        "date",
        "key",
        "local_path",
        "size",
        "row_count",
        "sha256",
        "etag",
        "columns",
    )
    reference_fields = (
        "ticker",
        "event_type",
        "local_path",
        "payload_bytes",
        "rows",
        "sha256",
        "schema_version",
        "sample_start",
        "sample_end",
        "request",
    )
    daily = [
        {field: row.get(field) for field in daily_fields if field in row}
        for row in manifest.get("daily_files", [])
    ]
    references = [
        {field: row.get(field) for field in reference_fields if field in row}
        for row in manifest.get("reference_downloads", [])
    ]
    daily.sort(key=lambda row: row["date"])
    references.sort(key=lambda row: (row["ticker"], row["event_type"]))
    reference_complete = manifest.get("reference_status") == "complete"
    return {
        "schema_version": 2,
        "manifest_type": "market_input_manifest",
        "provider": manifest.get("provider"),
        "sample_start": manifest.get("sample_start"),
        "sample_end": manifest.get("sample_end"),
        "verification_status": "pending_independent_verification",
        "daily_file_count": len(daily),
        "daily_compressed_bytes": sum(int(row.get("size", 0)) for row in daily),
        "daily_files": daily,
        "reference_file_count": len(references),
        "reference_status": "complete" if reference_complete else "incomplete",
        "reference_files": references,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/data_config.toml"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument(
        "--max-files", type=int, help="Testing only: restrict chronological daily files"
    )
    parser.add_argument(
        "--public-manifest",
        type=Path,
        default=Path("data/manifests/market_input_manifest.json"),
        help="Sanitised, deterministic manifest written outside the ignored licensed-data tree",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)
    study = config["study"]
    massive = config["massive"]
    paths = config["paths"]
    start = dt.date.fromisoformat(study["sample_start"])
    end = dt.date.fromisoformat(study["sample_end"])
    client = make_s3_client(config)
    objects = iter_objects(
        client,
        massive["flat_file_bucket"],
        massive["daily_aggregate_prefix"],
        start,
        end,
    )
    if args.max_files is not None:
        objects = objects[: args.max_files]
    total_bytes = sum(item["size"] for item in objects)
    print(f"Daily files: {len(objects)} ({total_bytes / (1024**3):.2f} GiB compressed)", flush=True)
    if args.list_only:
        return
    if not objects:
        raise RuntimeError("No entitled daily files were found for the configured sample")

    raw_root = project_path(project_root, paths["licensed_market_data"])
    downloaded: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                client,
                massive["flat_file_bucket"],
                project_root,
                raw_root,
                item,
                args.overwrite,
            ): item
            for item in objects
        }
        for count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            downloaded.append(future.result())
            if count % 100 == 0 or count == len(futures):
                print(f"Daily downloads validated: {count}/{len(futures)}", flush=True)
    downloaded.sort(key=lambda row: row["date"])

    manifest = {
        "schema_version": 2,
        "provider": "Massive",
        "sample_start": start.isoformat(),
        "sample_end": end.isoformat(),
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "daily_file_count": len(downloaded),
        "daily_compressed_bytes": sum(row["size"] for row in downloaded),
        "daily_files": downloaded,
        "reference_downloads": [],
        "reference_file_count": 0,
        "reference_status": "pending" if not args.skip_reference else "skipped",
    }
    manifest_path = project_path(project_root, paths["download_manifest_json"])
    write_json_atomic(manifest_path, manifest)
    print(f"Daily manifest checkpoint written: {manifest_path}", flush=True)

    if not args.skip_reference:
        reference_records = download_references(
            config, project_root, start.isoformat(), end.isoformat(), args.overwrite
        )
        manifest["reference_downloads"] = reference_records
        manifest["reference_file_count"] = len(reference_records)
        manifest["reference_status"] = "complete"
        manifest["generated_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
        write_json_atomic(manifest_path, manifest)
    public_manifest_path = (
        args.public_manifest
        if args.public_manifest.is_absolute()
        else project_root / args.public_manifest
    )
    write_json_atomic(public_manifest_path, public_input_manifest(manifest))
    print(f"Manifest written: {manifest_path}", flush=True)
    print(f"Sanitised manifest written: {public_manifest_path}", flush=True)


if __name__ == "__main__":
    main()
