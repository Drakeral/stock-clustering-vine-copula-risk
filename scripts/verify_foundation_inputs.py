#!/usr/bin/env python3
"""Verify licensed market inputs independently of the provider file listing.

This command checks every file recorded in the operational download manifest,
recomputes hashes and row counts, and compares daily dates with an independent
XNYS full-session calendar. It emits a sanitised manifest outside ``data/raw``.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import gzip
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

try:
    from scripts.pipeline_io import write_json_atomic
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from pipeline_io import write_json_atomic


EXPECTED_DAILY_COLUMNS = {
    "ticker",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "window_start",
    "transactions",
}

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

REQUIRED_DAILY_FIELDS = {
    "date",
    "key",
    "local_path",
    "size",
    "row_count",
    "sha256",
    "columns",
}

REQUIRED_REFERENCE_FIELDS = {
    "ticker",
    "event_type",
    "local_path",
    "payload_bytes",
    "rows",
    "sha256",
    "schema_version",
    "sample_start",
    "sample_end",
}

# Full-day closures not represented by recurring XNYS holiday rules.
SPECIAL_CLOSURES = {
    dt.date(2018, 12, 5): "National Day of Mourning for President George H. W. Bush",
    dt.date(2025, 1, 9): "National Day of Mourning for President Jimmy Carter",
}


@dataclass(frozen=True)
class FrozenInputSpec:
    """Expected input identity taken from version-controlled configuration."""

    provider: str
    manifest_schema_version: int
    sample_start: str
    sample_end: str
    daily_aggregate_prefix: str
    licensed_market_data: str

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        provider: str = "Massive",
        manifest_schema_version: int = 2,
    ) -> FrozenInputSpec:
        return cls(
            provider=provider,
            manifest_schema_version=manifest_schema_version,
            sample_start=str(config["study"]["sample_start"]),
            sample_end=str(config["study"]["sample_end"]),
            daily_aggregate_prefix=str(config["massive"]["daily_aggregate_prefix"]),
            licensed_market_data=str(config["paths"]["licensed_market_data"]),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """Return the exact deterministic JSON representation written by this script."""

    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> dt.date:
    matches = [
        day
        for day in calendar.Calendar().itermonthdates(year, month)
        if day.month == month and day.weekday() == weekday
    ]
    return matches[occurrence - 1]


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    matches = [
        day
        for day in calendar.Calendar().itermonthdates(year, month)
        if day.month == month and day.weekday() == weekday
    ]
    return matches[-1]


def _easter_sunday(year: int) -> dt.date:
    """Gregorian Easter via the anonymous computus algorithm."""

    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return dt.date(year, month, day)


def _observed(date: dt.date, observe_saturday_on_friday: bool = True) -> dt.date | None:
    if date.weekday() == calendar.SATURDAY:
        return date - dt.timedelta(days=1) if observe_saturday_on_friday else None
    if date.weekday() == calendar.SUNDAY:
        return date + dt.timedelta(days=1)
    return date


def xnys_holidays(year: int) -> set[dt.date]:
    # XNYS does not close on the preceding Friday when New Year's Day is Saturday.
    holidays = {
        _observed(dt.date(year, 1, 1), observe_saturday_on_friday=False),
        _nth_weekday(year, 1, calendar.MONDAY, 3),
        _nth_weekday(year, 2, calendar.MONDAY, 3),
        _easter_sunday(year) - dt.timedelta(days=2),
        _last_weekday(year, 5, calendar.MONDAY),
        _observed(dt.date(year, 7, 4)),
        _nth_weekday(year, 9, calendar.MONDAY, 1),
        _nth_weekday(year, 11, calendar.THURSDAY, 4),
        _observed(dt.date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(_observed(dt.date(year, 6, 19)))
    holidays.update(day for day in SPECIAL_CLOSURES if day.year == year)
    return {day for day in holidays if day is not None}


def xnys_sessions(start: dt.date, end: dt.date) -> list[dt.date]:
    if end < start:
        raise ValueError("XNYS calendar end precedes start")
    sessions: list[dt.date] = []
    day = start
    while day <= end:
        if day.weekday() < calendar.SATURDAY and day not in xnys_holidays(day.year):
            sessions.append(day)
        day += dt.timedelta(days=1)
    return sessions


def _safe_project_path(project_root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise ValueError(f"Manifest path must be project-relative: {relative}")
    resolved = (project_root / raw).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes project root: {relative}") from exc
    return resolved


def inspect_daily(
    path: Path,
    expected_date: dt.date,
    required_tickers: set[str],
) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        columns = handle.readline().strip().split(",")
        if len(columns) != len(set(columns)) or set(columns) != EXPECTED_DAILY_COLUMNS:
            raise ValueError(f"unexpected daily schema: {columns}")
        timestamp_index = columns.index("window_start")
        ticker_index = columns.index("ticker")
        row_count = 0
        content_dates: Counter[dt.date] = Counter()
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            values = line.rstrip("\r\n").split(",")
            if len(values) != len(columns):
                raise ValueError(f"daily row {line_number} has {len(values)} fields")
            try:
                timestamp_ns = int(values[timestamp_index])
                seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
                if nanoseconds < 0:
                    raise ValueError
                content_date = dt.datetime.fromtimestamp(seconds, tz=dt.UTC).date()
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"daily row {line_number} has invalid window_start") from exc
            ticker = values[ticker_index]
            if ticker in required_tickers and content_date != expected_date:
                raise ValueError(
                    f"daily row {line_number} ({ticker}) content date {content_date} "
                    f"differs from manifest date {expected_date}"
                )
            content_dates[content_date] += 1
            row_count += 1
    if row_count <= 0:
        raise ValueError("daily file has no data rows")
    modal_date, modal_count = content_dates.most_common(1)[0]
    if modal_date != expected_date:
        raise ValueError(
            f"modal daily content date {modal_date} differs from manifest date {expected_date}"
        )
    return {
        "columns": sorted(columns),
        "content_date": expected_date.isoformat(),
        "off_date_row_count": row_count - modal_count,
        "row_count": row_count,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def inspect_reference(
    path: Path,
    ticker: str,
    event_type: str,
    sample_start: str,
    sample_end: str,
    provider: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported reference schema")
    if payload.get("provider") != provider:
        raise ValueError("reference provider differs from frozen configuration")
    if payload.get("ticker") != ticker or payload.get("event_type") != event_type:
        raise ValueError("reference identity differs from manifest")
    if payload.get("sample_start") != sample_start or payload.get("sample_end") != sample_end:
        raise ValueError("reference sample dates differ from frozen configuration")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("reference results are not a list")
    _validate_reference_event_dates(results, event_type, sample_start, sample_end)
    return {
        "schema_version": payload["schema_version"],
        "rows": len(results),
        "payload_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "sample_start": payload.get("sample_start"),
        "sample_end": payload.get("sample_end"),
    }


def _validate_reference_event_dates(
    results: list[object],
    event_type: str,
    sample_start: str,
    sample_end: str,
) -> None:
    """Validate event-date presence, range, and ordering for one reference payload."""

    date_field = "execution_date" if event_type == "splits" else "ex_dividend_date"
    if any(not isinstance(row, dict) or not row.get(date_field) for row in results):
        raise ValueError(f"reference result lacks {date_field}")

    lower = dt.date.fromisoformat(sample_start)
    upper = dt.date.fromisoformat(sample_end)
    event_dates: list[dt.date] = []
    for index, row in enumerate(results):
        row_dict = cast(dict[str, Any], row)
        try:
            event_date = dt.date.fromisoformat(str(row_dict[date_field]))
        except ValueError as exc:
            raise ValueError(f"reference result {index} has invalid {date_field}") from exc
        if not lower <= event_date <= upper:
            raise ValueError(f"reference result {index} is outside the frozen sample")
        event_dates.append(event_date)
    if event_dates != sorted(event_dates):
        raise ValueError(f"reference results are not sorted by {date_field}")


def expected_reference_keys(
    universe_path: Path, security_master_path: Path
) -> set[tuple[str, str]]:
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    master = json.loads(security_master_path.read_text(encoding="utf-8"))
    tickers = {row["ticker"] for row in universe["constituents"]}
    for segments in master["mappings"].values():
        tickers.update(segment["ticker"] for segment in segments)
    return {(ticker, event_type) for ticker in tickers for event_type in ("splits", "dividends")}


def _missing_fields(record: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(field for field in required if field not in record or record[field] is None)


def _canonical_daily_identity(spec: FrozenInputSpec, day: dt.date) -> tuple[str, str]:
    filename = f"{day.isoformat()}.csv.gz"
    key = f"{spec.daily_aggregate_prefix.rstrip('/')}/{day:%Y}/{day:%m}/{filename}"
    local_path = (
        Path(spec.licensed_market_data) / "day_aggs" / f"{day:%Y}" / f"{day:%m}" / filename
    ).as_posix()
    return key, local_path


def enrich_legacy_operational_manifest(
    project_root: Path,
    manifest: dict[str, Any],
    reference_root: Path,
) -> dict[str, Any]:
    """Add independently observed fields required by the strict v2 verifier."""

    if manifest.get("schema_version") != 1:
        return manifest
    upgraded = {**manifest, "schema_version": 2}
    daily_value = manifest.get("daily_files")
    if not isinstance(daily_value, list):
        raise ValueError("Legacy operational manifest daily_files must be a list")
    daily_records: list[dict[str, Any]] = []
    for original in daily_value:
        if not isinstance(original, dict):
            raise ValueError("Legacy operational daily record must be an object")
        record = dict(original)
        try:
            day = dt.date.fromisoformat(str(record["date"]))
            path = _safe_project_path(project_root, str(record["local_path"]))
            observed = inspect_daily(path, day, set())
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(
                f"Cannot enrich legacy daily record {record.get('date', 'unknown')}: {exc}"
            ) from exc
        record["row_count"] = observed["row_count"]
        record["columns"] = observed["columns"]
        daily_records.append(record)
    upgraded["daily_files"] = daily_records

    reference_value = manifest.get("reference_downloads")
    if not isinstance(reference_value, list):
        raise ValueError("Legacy operational manifest reference_downloads must be a list")
    reference_records: list[dict[str, Any]] = []
    for original in reference_value:
        if not isinstance(original, dict):
            raise ValueError("Legacy operational reference record must be an object")
        record = dict(original)
        ticker = str(record.get("ticker", ""))
        event_type = str(record.get("event_type", ""))
        inferred = reference_root / event_type / f"{ticker.replace('.', '_')}.json"
        try:
            record["local_path"] = inferred.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("Legacy reference path is outside the project") from exc
        reference_records.append(record)
    upgraded["reference_downloads"] = reference_records
    upgraded["reference_file_count"] = len(reference_records)
    return upgraded


def _record_list(
    manifest: dict[str, Any],
    primary_key: str,
    *,
    errors: list[dict[str, str]],
    scope: str,
    fallback_key: str | None = None,
) -> list[dict[str, Any]]:
    """Return one manifest record list while recording structural defects."""

    if fallback_key and primary_key in manifest and fallback_key in manifest:
        errors.append({"scope": scope, "reason": "ambiguous_reference_lists"})
    selected_key = primary_key if primary_key in manifest or fallback_key is None else fallback_key
    value = manifest.get(selected_key)
    if not isinstance(value, list):
        errors.append({"scope": scope, "reason": f"{selected_key}_not_list"})
        return []
    return value


def _require_integer_fields(record: dict[str, Any], fields: tuple[str, ...]) -> None:
    """Require genuine integers, rejecting booleans despite their integer subtype."""

    for field in fields:
        if not isinstance(record[field], int) or isinstance(record[field], bool):
            raise ValueError(f"{field} is not an integer")


def _require_sha256(record: dict[str, Any]) -> None:
    """Require the canonical lowercase SHA-256 representation."""

    value = record["sha256"]
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError("sha256 is not a lowercase hexadecimal digest")


def _require_operational_status(record: dict[str, Any]) -> None:
    """Reject operational states outside the frozen download vocabulary."""

    status = record.get("status")
    if status not in {None, "existing", "downloaded", "verified"}:
        raise ValueError(f"invalid operational status: {status}")


def _validate_daily_record(
    project_root: Path,
    record: dict[str, Any],
    *,
    frozen: FrozenInputSpec,
    required_tickers: set[str],
) -> tuple[dt.date, dict[str, Any]]:
    """Validate one daily manifest record and return its observed public fields."""

    missing_fields = _missing_fields(record, REQUIRED_DAILY_FIELDS)
    if missing_fields:
        raise ValueError(f"missing required fields: {','.join(missing_fields)}")
    day = dt.date.fromisoformat(str(record["date"]))
    canonical_key, canonical_path = _canonical_daily_identity(frozen, day)
    if record["key"] != canonical_key:
        raise ValueError("key differs from canonical provider key")
    if record["local_path"] != canonical_path:
        raise ValueError("local_path differs from canonical configured path")
    _require_integer_fields(record, ("size", "row_count"))
    _require_sha256(record)
    if record["columns"] != sorted(EXPECTED_DAILY_COLUMNS):
        raise ValueError("columns differ from frozen daily schema")

    path = _safe_project_path(project_root, str(record["local_path"]))
    if not path.is_file():
        raise ValueError("file is absent")
    observed = inspect_daily(path, day, required_tickers)
    for field in ("size", "row_count", "sha256", "columns"):
        if record[field] != observed[field]:
            raise ValueError(f"{field} differs from manifest")
    _require_operational_status(record)
    verified = {
        "date": day.isoformat(),
        "key": record["key"],
        "local_path": record["local_path"],
        "etag": record.get("etag"),
        **observed,
    }
    return day, verified


def _verify_daily_records(
    project_root: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    frozen: FrozenInputSpec,
    required_tickers: set[str],
    errors: list[dict[str, str]],
) -> tuple[list[dt.date], list[dict[str, Any]]]:
    """Verify all daily records, collecting every independently observable defect."""

    if manifest.get("daily_file_count") != len(records):
        errors.append({"scope": "daily_manifest", "reason": "daily_file_count_mismatch"})
    recorded_dates: list[dt.date] = []
    verified: list[dict[str, Any]] = []
    declared_bytes = 0
    for record in records:
        if not isinstance(record, dict):
            errors.append({"scope": "daily:unknown", "reason": "record_not_object"})
            continue
        date_text = str(record.get("date", ""))
        try:
            day, observed = _validate_daily_record(
                project_root,
                record,
                frozen=frozen,
                required_tickers=required_tickers,
            )
            recorded_dates.append(day)
            verified.append(observed)
            declared_bytes += int(record["size"])
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"scope": f"daily:{date_text or 'unknown'}", "reason": str(exc)})
    if manifest.get("daily_compressed_bytes") != declared_bytes:
        errors.append({"scope": "daily_manifest", "reason": "daily_compressed_bytes_mismatch"})
    return recorded_dates, verified


def _append_calendar_errors(
    recorded_dates: list[dt.date],
    expected_dates: list[dt.date],
    errors: list[dict[str, str]],
) -> None:
    """Record duplicate, missing, and non-session dates without short-circuiting."""

    if len(recorded_dates) != len(set(recorded_dates)):
        errors.append({"scope": "daily_calendar", "reason": "duplicate_dates"})
    expected_set, recorded_set = set(expected_dates), set(recorded_dates)
    errors.extend(
        {"scope": "daily_calendar", "reason": f"missing_xnys_session:{day}"}
        for day in sorted(expected_set - recorded_set)
    )
    errors.extend(
        {"scope": "daily_calendar", "reason": f"non_xnys_date:{day}"}
        for day in sorted(recorded_set - expected_set)
    )


def _append_reference_manifest_errors(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    expected_keys: set[tuple[str, str]],
    errors: list[dict[str, str]],
) -> None:
    """Validate reference-list completeness and uniqueness."""

    sequence = [
        (str(record.get("ticker")), str(record.get("event_type")))
        for record in records
        if isinstance(record, dict)
    ]
    recorded_keys = set(sequence)
    for ticker, event_type in sorted(key for key, count in Counter(sequence).items() if count > 1):
        errors.append(
            {"scope": "reference_manifest", "reason": f"duplicate_identity:{ticker}:{event_type}"}
        )
    checks = (
        (manifest.get("reference_status") == "complete", "reference_status_not_complete"),
        (manifest.get("reference_file_count") == len(records), "reference_file_count_mismatch"),
        (len(records) == len(expected_keys), "reference_count_mismatch"),
    )
    errors.extend(
        {"scope": "reference_manifest", "reason": reason}
        for condition, reason in checks
        if not condition
    )
    errors.extend(
        {"scope": "reference_manifest", "reason": f"missing_record:{ticker}:{event_type}"}
        for ticker, event_type in sorted(expected_keys - recorded_keys)
    )
    errors.extend(
        {"scope": "reference_manifest", "reason": f"unexpected_record:{ticker}:{event_type}"}
        for ticker, event_type in sorted(recorded_keys - expected_keys)
    )


def _validate_reference_record(
    project_root: Path,
    reference_root: Path,
    record: dict[str, Any],
    *,
    frozen: FrozenInputSpec,
) -> dict[str, Any]:
    """Validate one corporate-action reference record and return public fields."""

    missing_fields = _missing_fields(record, REQUIRED_REFERENCE_FIELDS)
    if missing_fields:
        raise ValueError(f"missing required fields: {','.join(missing_fields)}")
    ticker = str(record["ticker"])
    event_type = str(record["event_type"])
    inferred = reference_root / event_type / f"{ticker.replace('.', '_')}.json"
    canonical_relative = inferred.resolve().relative_to(project_root.resolve()).as_posix()
    if record["local_path"] != canonical_relative:
        raise ValueError("local_path differs from canonical reference identity")
    _require_integer_fields(record, ("payload_bytes", "rows"))
    _require_sha256(record)
    expected_values = {
        "schema_version": 1,
        "sample_start": frozen.sample_start,
        "sample_end": frozen.sample_end,
    }
    for field, expected in expected_values.items():
        if record[field] != expected:
            raise ValueError(f"{field} differs from frozen reference configuration")

    path = _safe_project_path(project_root, str(record["local_path"]))
    if not path.is_file():
        raise ValueError("file is absent")
    observed = inspect_reference(
        path,
        ticker,
        event_type,
        frozen.sample_start,
        frozen.sample_end,
        frozen.provider,
    )
    for field in ("payload_bytes", "rows", "sha256", *expected_values):
        if record[field] != observed[field]:
            raise ValueError(f"{field} differs from manifest")
    _require_operational_status(record)
    return {
        "ticker": ticker,
        "event_type": event_type,
        "local_path": path.relative_to(project_root.resolve()).as_posix(),
        "request": record.get("request"),
        **observed,
    }


def _verify_reference_records(
    project_root: Path,
    reference_root: Path,
    records: list[dict[str, Any]],
    *,
    frozen: FrozenInputSpec,
    errors: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Verify corporate-action payload records while retaining all discovered errors."""

    verified: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            errors.append({"scope": "reference:unknown:unknown", "reason": "record_not_object"})
            continue
        ticker = str(record.get("ticker", ""))
        event_type = str(record.get("event_type", ""))
        try:
            verified.append(
                _validate_reference_record(
                    project_root,
                    reference_root,
                    record,
                    frozen=frozen,
                )
            )
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"scope": f"reference:{ticker}:{event_type}", "reason": str(exc)})
    return verified


def verify_foundation_inputs(
    project_root: Path,
    manifest: dict[str, Any],
    universe_path: Path,
    security_master_path: Path,
    reference_root: Path,
    frozen: FrozenInputSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify operational inputs and return a sanitised manifest plus gate audit."""

    errors: list[dict[str, str]] = []
    expected_metadata = {
        "schema_version": frozen.manifest_schema_version,
        "provider": frozen.provider,
        "sample_start": frozen.sample_start,
        "sample_end": frozen.sample_end,
    }
    for field, expected in expected_metadata.items():
        if manifest.get(field) != expected:
            errors.append(
                {
                    "scope": "manifest_identity",
                    "reason": f"{field}_mismatch:expected={expected}:observed={manifest.get(field)}",
                }
            )

    daily_records = _record_list(
        manifest,
        "daily_files",
        errors=errors,
        scope="daily_manifest",
    )
    reference_records = _record_list(
        manifest,
        "reference_downloads",
        fallback_key="reference_files",
        errors=errors,
        scope="reference_manifest",
    )

    start = dt.date.fromisoformat(frozen.sample_start)
    end = dt.date.fromisoformat(frozen.sample_end)
    expected_dates = xnys_sessions(start, end)
    expected_references = expected_reference_keys(universe_path, security_master_path)
    required_tickers = {ticker for ticker, _ in expected_references}

    recorded_dates, verified_daily = _verify_daily_records(
        project_root,
        manifest,
        daily_records,
        frozen=frozen,
        required_tickers=required_tickers,
        errors=errors,
    )
    _append_calendar_errors(recorded_dates, expected_dates, errors)
    _append_reference_manifest_errors(manifest, reference_records, expected_references, errors)
    verified_references = _verify_reference_records(
        project_root,
        reference_root,
        reference_records,
        frozen=frozen,
        errors=errors,
    )

    verified_daily.sort(key=lambda row: row["date"])
    verified_references.sort(key=lambda row: (row["ticker"], row["event_type"]))
    status = "pass" if not errors else "blocked"
    public_manifest = {
        "schema_version": 2,
        "manifest_type": "market_input_manifest",
        "provider": frozen.provider,
        "sample_start": frozen.sample_start,
        "sample_end": frozen.sample_end,
        "verification_status": status,
        "calendar": {
            "name": "XNYS",
            "implementation": "deterministic recurring-holiday rules plus enumerated full-day closures",
            "expected_session_count": len(expected_dates),
            "special_closures": {
                day.isoformat(): reason
                for day, reason in SPECIAL_CLOSURES.items()
                if start <= day <= end
            },
        },
        "daily_file_count": len(verified_daily),
        "daily_compressed_bytes": sum(row["size"] for row in verified_daily),
        "daily_files": verified_daily,
        "reference_file_count": len(verified_references),
        "reference_status": "complete"
        if len(verified_references) == len(expected_references)
        else "incomplete",
        "reference_files": verified_references,
    }
    audit = {
        "schema_version": 2,
        "gate": "market_input_integrity",
        "status": status,
        "public_manifest_sha256": hashlib.sha256(canonical_json_bytes(public_manifest)).hexdigest(),
        "checks": {
            "expected_xnys_sessions": len(expected_dates),
            "recorded_daily_files": len(daily_records),
            "verified_daily_files": len(verified_daily),
            "expected_reference_files": len(expected_references),
            "recorded_reference_files": len(reference_records),
            "verified_reference_files": len(verified_references),
        },
        "errors": errors,
    }
    return public_manifest, audit


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/data_config.toml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--public-manifest-output",
        type=Path,
        default=Path("data/manifests/market_input_manifest.json"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/audit/market_input_integrity.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)
    paths = config["paths"]
    manifest_path = args.manifest or Path(paths["download_manifest_json"])
    manifest_path = manifest_path if manifest_path.is_absolute() else project_root / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = enrich_legacy_operational_manifest(
        project_root,
        manifest,
        _safe_project_path(project_root, paths["corporate_actions_directory"]),
    )
    public, audit = verify_foundation_inputs(
        project_root,
        manifest,
        _safe_project_path(project_root, paths["universe_json"]),
        _safe_project_path(project_root, paths["security_master_json"]),
        _safe_project_path(project_root, paths["corporate_actions_directory"]),
        FrozenInputSpec.from_config(config),
    )
    public_path = (
        args.public_manifest_output
        if args.public_manifest_output.is_absolute()
        else project_root / args.public_manifest_output
    )
    audit_path = (
        args.audit_output if args.audit_output.is_absolute() else project_root / args.audit_output
    )
    write_json_atomic(audit_path, audit)
    if audit["status"] == "pass":
        write_json_atomic(public_path, public)
    print(f"market_input_integrity={audit['status']}")
    print(f"verified daily={public['daily_file_count']} reference={public['reference_file_count']}")
    if audit["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
