#!/usr/bin/env python3
"""Construct the fixed-universe total-return panel and coverage audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9-3.10
    import tomli as tomllib

try:
    from scripts.pipeline_io import (
        write_json_atomic,
        write_parquet_atomic,
        write_text_atomic,
    )
except ModuleNotFoundError:  # Support direct execution as ``python scripts/...``.
    from pipeline_io import write_json_atomic, write_parquet_atomic, write_text_atomic


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def project_path(project_root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else project_root / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


ALLOWED_MANUAL_ACTION_TYPES = {
    "literal_split",
    "stock_dividend",
    "spin_off",
    "merger_exchange",
    "cash_acquisition",
}


def optional_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    """Load an optional versioned control file without weakening its validation."""
    if not path.exists():
        return default
    payload = load_json(path)
    if "schema_version" not in payload:
        raise ValueError(f"Versioned control file has no schema_version: {path}")
    return payload


def build_segments(
    constituents: list[dict[str, Any]],
    master: dict[str, Any],
    start: str,
    end: str,
) -> pd.DataFrame:
    rows = []
    configured = master.get("mappings", {})
    for constituent in constituents:
        research_ticker = constituent["ticker"]
        segments = configured.get(
            research_ticker,
            [{"ticker": research_ticker, "start": start, "end": end, "event": "unchanged_ticker"}],
        )
        for segment in segments:
            rows.append(
                {
                    "research_ticker": research_ticker,
                    "provider_ticker": segment["ticker"],
                    "start": pd.Timestamp(segment["start"]),
                    "end": pd.Timestamp(segment["end"]),
                    "event": segment["event"],
                }
            )
    frame = pd.DataFrame(rows).sort_values(["provider_ticker", "research_ticker", "start"])
    for (provider_ticker, research_ticker), group in frame.groupby(
        ["provider_ticker", "research_ticker"]
    ):
        ordered = group.sort_values("start")
        prior_end: pd.Timestamp | None = None
        for row in ordered.itertuples(index=False):
            if prior_end is not None and row.start <= prior_end:
                raise ValueError(
                    f"Overlapping segments for research ticker {research_ticker} "
                    f"and provider ticker {provider_ticker}"
                )
            prior_end = row.end
    return frame


def map_provider_rows(frame: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(
            research_ticker=pd.Series(dtype="string"), continuity_event=pd.Series(dtype="string")
        )
    mapped = []
    for provider_ticker, rows in frame.groupby("ticker", sort=False):
        options = segments.loc[segments["provider_ticker"] == provider_ticker]
        for segment in options.itertuples(index=False):
            mask = rows["date"].between(segment.start, segment.end)
            if mask.any():
                selected = rows.loc[mask].copy()
                selected["research_ticker"] = segment.research_ticker
                selected["continuity_event"] = segment.event
                mapped.append(selected)
    if not mapped:
        return pd.DataFrame(columns=list(frame.columns) + ["research_ticker", "continuity_event"])
    result = pd.concat(mapped, ignore_index=True)
    duplicates = result.duplicated(["date", "research_ticker"], keep=False)
    if duplicates.any():
        sample = (
            result.loc[duplicates, ["date", "ticker", "research_ticker"]].head().to_dict("records")
        )
        raise ValueError(f"Security master maps multiple rows to one security-date: {sample}")
    return result


def ancillary_tickers(actions: dict[str, Any]) -> set[str]:
    return {
        distribution["provider_ticker"]
        for action in actions.get("actions", [])
        for distribution in action.get("distributed_securities", [])
    }


def add_synthetic_primary_rows(
    prices: pd.DataFrame,
    actions: dict[str, Any],
) -> pd.DataFrame:
    """Add explicitly configured non-market terminal-consideration observations."""
    rows: list[dict[str, Any]] = []
    keys = set(zip(prices["research_ticker"], pd.to_datetime(prices["date"]), strict=False))
    for action in actions.get("actions", []):
        synthetic = action.get("synthetic_primary_observation")
        if not synthetic:
            continue
        date = pd.Timestamp(action["effective_date"])
        key = (action["research_ticker"], date)
        if key in keys:
            raise ValueError(
                f"Synthetic primary observation overlaps an observed price: {key[0]} {date.date()}"
            )
        cash = float(action.get("cash_per_old_share", 0.0))
        contingent = float(action.get("contingent_value_per_old_share", 0.0))
        close = float(synthetic["close"])
        if synthetic.get("valuation_formula") == "cash_plus_contingent" and not math.isclose(
            close, cash + contingent, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"Synthetic close for {action['action_id']} does not equal cash plus contingent value"
            )
        rows.append(
            {
                "date": date,
                "research_ticker": action["research_ticker"],
                "ticker": action["primary_provider_ticker"],
                "continuity_event": "synthetic_terminal_consideration",
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 0,
                "transactions": 0,
                "window_start": int(date.value),
                "observation_status": "synthetic_terminal_consideration",
                "lifecycle_classification": "active_price",
                "is_synthetic": True,
            }
        )
        keys.add(key)
    if not rows:
        return prices
    return pd.concat([prices, pd.DataFrame(rows)], ignore_index=True, sort=False)


def read_daily_files(
    project_root: Path,
    manifest: dict[str, Any],
    segments: pd.DataFrame,
    additional_tickers: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    provider_tickers = set(segments["provider_ticker"])
    selected_tickers = provider_tickers | additional_tickers
    frames = []
    usecols = ["ticker", "volume", "open", "close", "high", "low", "window_start", "transactions"]
    for count, item in enumerate(manifest["daily_files"], start=1):
        path = project_path(project_root, item["local_path"])
        if not path.exists():
            raise FileNotFoundError(f"Manifested provider file is missing: {path}")
        expected_hash = item.get("sha256")
        if not expected_hash or sha256_file(path) != expected_hash:
            raise ValueError(f"Manifested provider file hash differs: {path}")
        daily = pd.read_csv(
            path,
            compression="gzip",
            usecols=usecols,
            dtype={"ticker": "string"},
        )
        daily = daily.loc[daily["ticker"].isin(selected_tickers)].copy()
        daily["date"] = pd.Timestamp(item["date"])
        if not daily.empty:
            frames.append(daily)
        if count % 250 == 0 or count == len(manifest["daily_files"]):
            print(f"Daily files processed: {count}/{len(manifest['daily_files'])}", flush=True)
    if not frames:
        raise RuntimeError("No candidate-universe price rows were found")
    raw = pd.concat(frames, ignore_index=True)
    mapped = map_provider_rows(raw.loc[raw["ticker"].isin(provider_tickers)].copy(), segments)
    ancillary = raw.loc[raw["ticker"].isin(additional_tickers), ["date", "ticker", "close"]].copy()
    ancillary["close"] = pd.to_numeric(ancillary["close"], errors="coerce")
    if ancillary.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate ancillary-security close detected")
    return mapped, ancillary


def verify_reference_files_against_manifest(
    project_root: Path,
    manifest: dict[str, Any],
    reference_root: Path,
) -> None:
    """Recheck every small reference payload against the independently verified manifest."""

    records = manifest.get("reference_files")
    if not isinstance(records, list) or not records:
        raise ValueError("Verified market manifest has no reference_files")
    if manifest.get("reference_file_count") != len(records):
        raise ValueError("Verified market manifest reference_file_count differs")
    identities: set[tuple[str, str]] = set()
    manifested_paths: set[Path] = set()
    for record in records:
        identity = (str(record.get("ticker", "")), str(record.get("event_type", "")))
        if identity in identities:
            raise ValueError(f"Duplicate verified reference identity: {identity}")
        identities.add(identity)
        path = project_path(project_root, str(record.get("local_path", "")))
        manifested_paths.add(path.resolve())
        expected_hash = record.get("sha256")
        if not path.is_file() or not expected_hash or sha256_file(path) != expected_hash:
            raise ValueError(f"Verified reference file is absent or has changed: {path}")
    if manifested_paths:
        actual_paths = {path.resolve() for path in reference_root.glob("*/*.json")}
        if actual_paths != manifested_paths:
            raise ValueError("Reference directory contents differ from the verified manifest")


def read_reference_events(reference_dir: Path, event_type: str) -> pd.DataFrame:
    rows = []
    for path in sorted((reference_dir / event_type).glob("*.json")):
        payload = load_json(path)
        for index, result in enumerate(payload.get("results", [])):
            rows.append(
                {
                    **result,
                    "_source_file": str(path),
                    "_source_index": index,
                    "_event_uid": result.get("id", f"{path.name}:{index}"),
                }
            )
    if not rows:
        if event_type == "splits":
            return pd.DataFrame(columns=["ticker", "execution_date", "split_from", "split_to"])
        return pd.DataFrame(columns=["ticker", "ex_dividend_date", "cash_amount"])
    return pd.DataFrame(rows)


def map_event_ticker(
    events: pd.DataFrame,
    date_field: str,
    segments: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return events.assign(
            research_ticker=pd.Series(dtype="string"), date=pd.Series(dtype="datetime64[ns]")
        )
    events = events.copy()
    events["date"] = pd.to_datetime(events[date_field])
    events = events.rename(columns={"ticker": "provider_event_ticker"})
    mapped = []
    for provider_ticker, rows in events.groupby("provider_event_ticker", sort=False):
        options = segments.loc[segments["provider_ticker"] == provider_ticker]
        for segment in options.itertuples(index=False):
            mask = rows["date"].between(segment.start, segment.end)
            if mask.any():
                selected = rows.loc[mask].copy()
                selected["research_ticker"] = segment.research_ticker
                mapped.append(selected)
    if not mapped:
        return pd.DataFrame(columns=list(events.columns) + ["research_ticker"])
    return pd.concat(mapped, ignore_index=True)


def aggregate_actions(
    reference_dir: Path,
    segments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_daily, dividend_daily, _ = aggregate_actions_v2(reference_dir, segments, {})
    return split_daily, dividend_daily


def aggregate_actions_v2(
    reference_dir: Path,
    segments: pd.DataFrame,
    manual_action_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Aggregate provider events and retain an event-level audit trail.

    Provider continuity factors that have an explicit manual reconstruction are
    excluded from arithmetic. Their original values remain in reconciliation
    records, preventing a spin-off from being applied twice.
    """
    splits = map_event_ticker(
        read_reference_events(reference_dir, "splits"), "execution_date", segments
    )
    dividends = map_event_ticker(
        read_reference_events(reference_dir, "dividends"), "ex_dividend_date", segments
    )

    manual_by_provider_event = {
        str(action["provider_event_id"]): action
        for action in manual_action_config.get("actions", [])
        if action.get("provider_event_id")
    }

    if splits.empty:
        split_daily = pd.DataFrame(columns=["research_ticker", "date", "split_factor"])
    else:
        splits["split_factor"] = pd.to_numeric(splits["split_to"]) / pd.to_numeric(
            splits["split_from"]
        )
        splits["manual_action_id"] = splits["_event_uid"].map(
            lambda event_id: manual_by_provider_event.get(str(event_id), {}).get("action_id")
        )
        applied_splits = splits.loc[splits["manual_action_id"].isna()].copy()
        split_daily = (
            applied_splits.groupby(["research_ticker", "date"], as_index=False)["split_factor"]
            .prod()
            .sort_values(["research_ticker", "date"])
        )

    if dividends.empty:
        dividend_daily = pd.DataFrame(columns=["research_ticker", "date", "cash_dividend"])
    else:
        dividends["cash_amount"] = pd.to_numeric(dividends["cash_amount"])
        dividend_daily = (
            dividends.groupby(["research_ticker", "date"], as_index=False)["cash_amount"]
            .sum()
            .rename(columns={"cash_amount": "cash_dividend"})
            .sort_values(["research_ticker", "date"])
        )

    records: list[dict[str, Any]] = []
    for event_type, raw, mapped in (
        ("split", read_reference_events(reference_dir, "splits"), splits),
        ("dividend", read_reference_events(reference_dir, "dividends"), dividends),
    ):
        if raw.empty:
            continue
        mapped_counts: dict[tuple[str, pd.Timestamp], int] = defaultdict(int)
        if not mapped.empty:
            for key, count in mapped.groupby(["research_ticker", "date"]).size().items():
                mapped_counts[(str(key[0]), pd.Timestamp(key[1]))] = int(count)
        for raw_row in raw.to_dict("records"):
            event_id = str(raw_row["_event_uid"])
            selected = (
                mapped.loc[mapped["_event_uid"].astype(str).eq(event_id)]
                if not mapped.empty
                else pd.DataFrame()
            )
            if selected.empty:
                records.append(
                    {
                        "event_id": event_id,
                        "event_type": event_type,
                        "provider_ticker": raw_row.get("ticker"),
                        "research_ticker": None,
                        "date": raw_row.get("execution_date") or raw_row.get("ex_dividend_date"),
                        "typed_event": "literal_split"
                        if event_type == "split"
                        else "cash_dividend",
                        "application_method": "not_applicable",
                        "aggregation_count": 0,
                        "mapping_status": "out_of_segment",
                    }
                )
                continue
            for mapped_row in selected.to_dict("records"):
                manual = manual_by_provider_event.get(event_id)
                typed_event = (
                    str(manual["event_type"])
                    if manual
                    else ("literal_split" if event_type == "split" else "cash_dividend")
                )
                record = {
                    "event_id": event_id,
                    "event_type": event_type,
                    "provider_ticker": mapped_row.get("provider_event_ticker"),
                    "research_ticker": mapped_row["research_ticker"],
                    "date": pd.Timestamp(mapped_row["date"]).strftime("%Y-%m-%d"),
                    "typed_event": typed_event,
                    "application_method": (
                        "manual_action_replacement" if manual else "provider_event_aggregation"
                    ),
                    "manual_action_id": manual.get("action_id") if manual else None,
                    "aggregation_count": mapped_counts[
                        (str(mapped_row["research_ticker"]), pd.Timestamp(mapped_row["date"]))
                    ],
                    "mapping_status": "mapped",
                }
                if event_type == "split":
                    record["provider_continuity_factor"] = float(mapped_row["split_factor"])
                else:
                    record["cash_amount"] = float(mapped_row["cash_amount"])
                records.append(record)
    return split_daily, dividend_daily, records


def reconcile_reference_events(
    records: list[dict[str, Any]],
    prices: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Assign the four frozen reconciliation dispositions to every provider event."""
    valid_keys = {
        (str(row.research_ticker), pd.Timestamp(row.date))
        for row in prices.loc[prices["close"].notna(), ["research_ticker", "date"]].itertuples(
            index=False
        )
    }
    reconciled = []
    for source in records:
        record = dict(source)
        if record["mapping_status"] == "out_of_segment":
            disposition = "out_of_segment"
        elif (str(record["research_ticker"]), pd.Timestamp(record["date"])) not in valid_keys:
            disposition = "unmatched_price"
        elif int(record["aggregation_count"]) > 1:
            disposition = "duplicate_aggregated"
        else:
            disposition = "applied"
        record["disposition"] = disposition
        reconciled.append(record)
    return reconciled


def manual_action_table(
    actions: dict[str, Any],
    ancillary_prices: pd.DataFrame,
) -> pd.DataFrame:
    price_lookup = ancillary_prices.set_index(["date", "ticker"])["close"]
    rows = []
    for action in actions.get("actions", []):
        event_type = str(action.get("event_type", ""))
        if event_type not in ALLOWED_MANUAL_ACTION_TYPES:
            raise ValueError(
                f"Manual action {action.get('action_id')} has unsupported event_type {event_type!r}"
            )
        action_date = pd.Timestamp(action["effective_date"])
        distribution_value = 0.0
        distribution_components = []
        for distribution in action.get("distributed_securities", []):
            key = (action_date, distribution["provider_ticker"])
            if key not in price_lookup.index:
                raise ValueError(
                    f"Missing closing price for distributed security {distribution['provider_ticker']} "
                    f"on {action['effective_date']}"
                )
            close = float(price_lookup.loc[key])
            ratio = float(distribution["shares_per_old_share"])
            distribution_value += close * ratio
            distribution_components.append(
                {
                    "provider_ticker": distribution["provider_ticker"],
                    "shares_per_old_share": ratio,
                    "close": close,
                    "value_per_old_share": close * ratio,
                }
            )
        rows.append(
            {
                "research_ticker": action["research_ticker"],
                "date": action_date,
                "manual_action_id": action["action_id"],
                "manual_action_type": event_type,
                "manual_primary_provider_ticker": action["primary_provider_ticker"],
                "manual_primary_share_multiplier": float(action["primary_shares_per_old_share"]),
                "manual_cash_per_old_share": float(action.get("cash_per_old_share", 0.0)),
                "manual_contingent_value_per_old_share": float(
                    action.get("contingent_value_per_old_share", 0.0)
                ),
                "manual_contingent_value_low": action.get("contingent_value_sensitivity", {}).get(
                    "low"
                ),
                "manual_contingent_value_high": action.get("contingent_value_sensitivity", {}).get(
                    "high"
                ),
                "stock_distribution_value": distribution_value,
                "stock_distribution_components": json.dumps(
                    distribution_components, sort_keys=True
                ),
                "provider_event_id": action.get("provider_event_id"),
                "provider_continuity_factor": action.get("provider_continuity_factor"),
                "manual_action_source_url": action["source_url"],
            }
        )
    return pd.DataFrame(rows)


def complete_lifecycle_grid(
    prices: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    segments: pd.DataFrame,
    lifecycle_config: dict[str, Any],
    exit_treatments: dict[str, Any],
    corporate_action_keys: set[tuple[str, pd.Timestamp]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Make every active security-session explicit and classify every absence.

    Verified halts receive a stale synthetic close and zero return. Unexplained
    holes receive a visible NaN placeholder, which both blocks the gate and
    prevents return construction from silently shifting across the gap.
    """
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if frame.duplicated(["research_ticker", "date"]).any():
        raise ValueError("Cannot complete lifecycle grid with duplicate security-date prices")
    for column, default in (
        ("observation_status", "observed"),
        ("lifecycle_classification", "active_price"),
        ("is_synthetic", False),
    ):
        if column not in frame:
            frame[column] = default
        else:
            frame[column] = frame[column].where(frame[column].notna(), default)
            if column == "is_synthetic":
                frame[column] = frame[column].astype(bool)

    security_config = lifecycle_config.get("securities", {})
    halt_lookup = {
        (str(item["research_ticker"]), pd.Timestamp(item["date"])): item
        for item in lifecycle_config.get("verified_halts", [])
    }
    configured_halts_used: set[tuple[str, pd.Timestamp]] = set()
    issues: list[str] = []
    missing_active_dates: list[dict[str, Any]] = []
    classification_counts: defaultdict[str, int] = defaultdict(int)
    output_rows: list[dict[str, Any]] = []

    trading_dates = pd.DatetimeIndex(sorted(pd.to_datetime(trading_dates).unique()))
    sample_end = pd.Timestamp(trading_dates.max())
    for ticker, ticker_segments in segments.groupby("research_ticker", sort=True):
        ticker = str(ticker)
        configured = security_config.get(ticker, {})
        listing_date = pd.Timestamp(configured.get("listing_date", ticker_segments["start"].min()))
        terminal_date = pd.Timestamp(configured.get("terminal_market_date", sample_end))
        treatment = exit_treatments.get(ticker, {})
        remove_value = treatment.get("remove_effective_date")
        remove_date = (
            pd.Timestamp(remove_value)
            if remove_value
            else pd.Timestamp(sample_end.date() + dt.timedelta(days=1))
        )
        cash_start = treatment.get("cash_carry_start")
        cash_start_date = pd.Timestamp(cash_start) if cash_start else None

        actual = {
            pd.Timestamp(row.date): row._asdict()
            for row in frame.loc[frame["research_ticker"].eq(ticker)].itertuples(index=False)
        }
        last_close: float | None = None
        for date in trading_dates:
            if date < listing_date:
                classification_counts["pre_inception"] += 1
                continue
            if date >= remove_date:
                classification_counts["removed"] += 1
                if date in actual:
                    issues.append(f"{ticker} {date.date()}: observed after removal date")
                continue
            if cash_start_date is not None and date >= cash_start_date:
                classification_counts["terminal_cash"] += 1
                if date in actual:
                    issues.append(
                        f"{ticker} {date.date()}: market row overlaps terminal-cash period"
                    )
                continue
            if date > terminal_date:
                classification_counts["unexplained_missing"] += 1
                missing_active_dates.append(
                    {
                        "research_ticker": ticker,
                        "date": date.strftime("%Y-%m-%d"),
                        "classification": "unexplained_missing",
                        "reason": "series ended without terminal-cash or removal classification",
                    }
                )
                issues.append(f"{ticker} {date.date()}: unexplained post-terminal absence")
                continue

            if date in actual:
                row = actual[date]
                row["observation_status"] = row.get("observation_status") or "observed"
                row["lifecycle_classification"] = "active_price"
                row["is_synthetic"] = bool(row.get("is_synthetic", False))
                output_rows.append(row)
                close = pd.to_numeric(pd.Series([row.get("close")]), errors="coerce").iloc[0]
                if pd.notna(close):
                    last_close = float(close)
                classification_counts[
                    "synthetic_terminal_consideration"
                    if row["observation_status"] == "synthetic_terminal_consideration"
                    else "observed"
                ] += 1
                continue

            segment_rows = ticker_segments.loc[
                ticker_segments["start"].le(date) & ticker_segments["end"].ge(date)
            ]
            halt = halt_lookup.get((ticker, date))
            has_action = (ticker, date) in corporate_action_keys
            if halt and not has_action and last_close is not None and len(segment_rows) == 1:
                segment = segment_rows.iloc[0]
                output_rows.append(
                    {
                        "date": date,
                        "research_ticker": ticker,
                        "ticker": segment["provider_ticker"],
                        "continuity_event": segment["event"],
                        "open": last_close,
                        "high": last_close,
                        "low": last_close,
                        "close": last_close,
                        "volume": 0,
                        "transactions": 0,
                        "window_start": int(date.value),
                        "observation_status": "verified_halt",
                        "lifecycle_classification": "verified_halt",
                        "is_synthetic": True,
                    }
                )
                configured_halts_used.add((ticker, date))
                classification_counts["verified_halt"] += 1
                missing_active_dates.append(
                    {
                        "research_ticker": ticker,
                        "date": date.strftime("%Y-%m-%d"),
                        "classification": "verified_halt",
                        "reason": halt["reason"],
                    }
                )
                continue

            reason = "unconfigured active-session absence"
            if halt and has_action:
                reason = "verified halt overlaps a corporate action and cannot be stale-filled"
            elif len(segment_rows) != 1:
                reason = "no unique provider segment covers active session"
            provider_ticker = (
                segment_rows.iloc[0]["provider_ticker"] if len(segment_rows) == 1 else pd.NA
            )
            continuity = segment_rows.iloc[0]["event"] if len(segment_rows) == 1 else "segment_gap"
            output_rows.append(
                {
                    "date": date,
                    "research_ticker": ticker,
                    "ticker": provider_ticker,
                    "continuity_event": continuity,
                    "open": np.nan,
                    "high": np.nan,
                    "low": np.nan,
                    "close": np.nan,
                    "volume": np.nan,
                    "transactions": np.nan,
                    "window_start": np.nan,
                    "observation_status": "unexplained_missing",
                    "lifecycle_classification": "unexplained_missing",
                    "is_synthetic": True,
                }
            )
            classification_counts["unexplained_missing"] += 1
            missing_active_dates.append(
                {
                    "research_ticker": ticker,
                    "date": date.strftime("%Y-%m-%d"),
                    "classification": "unexplained_missing",
                    "reason": reason,
                    "corporate_action_on_missing_date": has_action,
                }
            )
            issues.append(f"{ticker} {date.date()}: {reason}")

    unused_halts = sorted(
        f"{ticker} {date.strftime('%Y-%m-%d')}"
        for ticker, date in set(halt_lookup) - configured_halts_used
    )
    issues.extend(f"Configured halt was not used: {value}" for value in unused_halts)
    completed = pd.DataFrame(output_rows)
    completed["date"] = pd.to_datetime(completed["date"])
    return completed.sort_values(["research_ticker", "date"]).reset_index(drop=True), {
        "schema_version": 2,
        "classification_counts": dict(sorted(classification_counts.items())),
        "missing_active_dates": missing_active_dates,
        "verified_halts_configured": len(halt_lookup),
        "verified_halts_applied": len(configured_halts_used),
        "issues": issues,
    }


def construct_returns(
    prices: pd.DataFrame,
    split_daily: pd.DataFrame,
    dividend_daily: pd.DataFrame,
    manual_actions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = prices.copy()
    for column, default in (
        ("observation_status", "observed"),
        ("lifecycle_classification", "active_price"),
        ("is_synthetic", False),
    ):
        if column not in frame:
            frame[column] = default
        else:
            frame[column] = frame[column].where(frame[column].notna(), default)
            if column == "is_synthetic":
                frame[column] = frame[column].astype(bool)
    numeric = ["volume", "open", "close", "high", "low", "window_start", "transactions"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["research_ticker", "date"])
    frame = frame.merge(split_daily, how="left", on=["research_ticker", "date"])
    frame = frame.merge(dividend_daily, how="left", on=["research_ticker", "date"])
    if manual_actions is None or manual_actions.empty:
        manual_actions = pd.DataFrame(
            columns=[
                "research_ticker",
                "date",
                "manual_action_id",
                "manual_action_type",
                "manual_primary_provider_ticker",
                "manual_primary_share_multiplier",
                "manual_cash_per_old_share",
                "manual_contingent_value_per_old_share",
                "manual_contingent_value_low",
                "manual_contingent_value_high",
                "stock_distribution_value",
                "stock_distribution_components",
                "provider_event_id",
                "provider_continuity_factor",
                "manual_action_source_url",
            ]
        )
    else:
        manual_actions = manual_actions.copy()
        manual_defaults = {
            "manual_action_type": pd.NA,
            "manual_contingent_value_per_old_share": 0.0,
            "manual_contingent_value_low": np.nan,
            "manual_contingent_value_high": np.nan,
            "provider_event_id": pd.NA,
            "provider_continuity_factor": np.nan,
        }
        for column, default in manual_defaults.items():
            if column not in manual_actions:
                manual_actions[column] = default
    frame = frame.merge(manual_actions, how="left", on=["research_ticker", "date"])
    wrong_primary = frame["manual_action_id"].notna() & frame["ticker"].ne(
        frame["manual_primary_provider_ticker"]
    )
    if wrong_primary.any():
        sample = frame.loc[
            wrong_primary,
            [
                "date",
                "research_ticker",
                "ticker",
                "manual_primary_provider_ticker",
                "manual_action_id",
            ],
        ].to_dict("records")
        raise ValueError(f"Manual action primary ticker mismatch: {sample}")
    for column in (
        "split_factor",
        "cash_dividend",
        "manual_primary_share_multiplier",
        "manual_cash_per_old_share",
        "manual_contingent_value_per_old_share",
        "manual_contingent_value_low",
        "manual_contingent_value_high",
        "stock_distribution_value",
        "provider_continuity_factor",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["split_factor"] = frame["split_factor"].fillna(1.0)
    frame["cash_dividend"] = frame["cash_dividend"].fillna(0.0)
    frame["manual_cash_per_old_share"] = frame["manual_cash_per_old_share"].fillna(0.0)
    frame["manual_contingent_value_per_old_share"] = frame[
        "manual_contingent_value_per_old_share"
    ].fillna(0.0)
    frame["stock_distribution_value"] = frame["stock_distribution_value"].fillna(0.0)
    frame["effective_share_multiplier"] = frame["manual_primary_share_multiplier"].fillna(
        frame["split_factor"]
    )
    frame["previous_close"] = frame.groupby("research_ticker")["close"].shift(1)
    frame["previous_provider_ticker"] = frame.groupby("research_ticker")["ticker"].shift(1)
    frame["previous_observation_status"] = frame.groupby("research_ticker")[
        "observation_status"
    ].shift(1)
    frame["ticker_segment_transition"] = frame["previous_provider_ticker"].notna() & frame[
        "ticker"
    ].ne(frame["previous_provider_ticker"])
    frame["price_gross_return"] = frame["close"] / frame["previous_close"]
    frame["total_consideration_per_old_share"] = (
        frame["effective_share_multiplier"] * frame["close"]
        + frame["split_factor"] * frame["cash_dividend"]
        + frame["manual_cash_per_old_share"]
        + frame["manual_contingent_value_per_old_share"]
        + frame["stock_distribution_value"]
    )
    frame["total_gross_return"] = (
        frame["total_consideration_per_old_share"] / frame["previous_close"]
    )
    invalid = (
        frame["previous_close"].isna()
        | frame["close"].le(0)
        | frame["previous_close"].le(0)
        | frame["total_gross_return"].le(0)
    )
    frame.loc[invalid, "total_gross_return"] = np.nan
    frame["total_return"] = frame["total_gross_return"] - 1.0
    frame["log_total_return"] = np.log(frame["total_gross_return"])
    frame["quality_flag"] = ""
    frame.loc[frame["close"].le(0), "quality_flag"] += "nonpositive_close;"
    frame.loc[frame["ticker_segment_transition"], "quality_flag"] += "ticker_segment_transition;"
    frame.loc[frame["manual_action_id"].notna(), "quality_flag"] += "manual_corporate_action;"
    frame.loc[frame["observation_status"].eq("verified_halt"), "quality_flag"] += (
        "verified_halt_zero_return;"
    )
    frame.loc[
        frame["previous_observation_status"].eq("verified_halt")
        & frame["observation_status"].eq("observed"),
        "quality_flag",
    ] += "gap_bridge_return;"
    frame.loc[
        frame["observation_status"].eq("synthetic_terminal_consideration"), "quality_flag"
    ] += "synthetic_terminal_consideration;"
    frame.loc[frame["observation_status"].eq("unexplained_missing"), "quality_flag"] += (
        "unexplained_missing_active_day;"
    )
    frame.loc[frame["total_return"].abs().gt(0.40), "quality_flag"] += "extreme_total_return;"
    frame.loc[
        frame["price_gross_return"].sub(1).abs().gt(0.40)
        & frame["split_factor"].eq(1.0)
        & frame["manual_action_id"].isna(),
        "quality_flag",
    ] += "extreme_unadjusted_return_without_split;"
    columns = [
        "date",
        "research_ticker",
        "ticker",
        "continuity_event",
        "observation_status",
        "lifecycle_classification",
        "is_synthetic",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "transactions",
        "window_start",
        "previous_close",
        "split_factor",
        "cash_dividend",
        "effective_share_multiplier",
        "manual_cash_per_old_share",
        "manual_contingent_value_per_old_share",
        "manual_contingent_value_low",
        "manual_contingent_value_high",
        "stock_distribution_value",
        "stock_distribution_components",
        "manual_action_id",
        "manual_action_type",
        "provider_event_id",
        "provider_continuity_factor",
        "manual_action_source_url",
        "total_consideration_per_old_share",
        "price_gross_return",
        "total_gross_return",
        "total_return",
        "log_total_return",
        "ticker_segment_transition",
        "previous_observation_status",
        "quality_flag",
    ]
    return frame[columns].sort_values(["date", "research_ticker"]).reset_index(drop=True)


def coverage_table(
    daily: pd.DataFrame,
    constituents: list[dict[str, Any]],
    trading_dates: pd.DatetimeIndex,
    training_start: pd.Timestamp,
    training_end: pd.Timestamp,
    minimum_coverage: float,
) -> pd.DataFrame:
    expected_dates = trading_dates[
        (trading_dates >= training_start) & (trading_dates <= training_end)
    ]
    expected_prices = len(expected_dates)
    expected_returns = max(0, expected_prices - 1)
    rows = []
    for constituent in constituents:
        ticker = constituent["ticker"]
        sample = daily.loc[
            daily["research_ticker"].eq(ticker)
            & daily["date"].between(training_start, training_end)
        ]
        price_count = int(sample["close"].notna().sum())
        return_count = int(sample["log_total_return"].notna().sum())
        price_coverage = price_count / expected_prices if expected_prices else math.nan
        return_coverage = return_count / expected_returns if expected_returns else math.nan
        include = bool(price_coverage >= minimum_coverage and return_coverage >= minimum_coverage)
        rows.append(
            {
                "ticker": ticker,
                "company_name": constituent["company_name"],
                "gics_sector": constituent["gics_sector"],
                "expected_price_days": expected_prices,
                "observed_price_days": price_count,
                "price_coverage": price_coverage,
                "expected_return_days": expected_returns,
                "observed_return_days": return_count,
                "return_coverage": return_coverage,
                "included": include,
                "exclusion_reason": "" if include else "initial_training_coverage_below_threshold",
            }
        )
    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def audit_exit_treatments(
    daily: pd.DataFrame,
    included: list[str],
    sample_end: pd.Timestamp,
    exit_treatments: dict[str, Any],
) -> dict[str, Any]:
    observed_last = (
        daily.loc[daily["research_ticker"].isin(included)]
        .groupby("research_ticker")["date"]
        .max()
        .to_dict()
    )
    early_endings = {
        ticker: date.strftime("%Y-%m-%d")
        for ticker, date in observed_last.items()
        if date < sample_end
    }
    issues = []
    for ticker, observed_date in early_endings.items():
        treatment = exit_treatments.get(ticker)
        if treatment is None:
            issues.append(f"{ticker}: early ending {observed_date} has no exit treatment")
            continue
        configured_date = treatment["last_market_return_date"]
        if configured_date != observed_date:
            issues.append(
                f"{ticker}: observed last return {observed_date} differs from configured {configured_date}"
            )
    for ticker in sorted(set(exit_treatments) - set(early_endings)):
        issues.append(
            f"{ticker}: exit treatment exists but the series does not end before sample end"
        )
    return {
        "early_endings": early_endings,
        "documented_exit_tickers": sorted(exit_treatments),
        "issues": issues,
    }


def build_portfolio_panel(
    stock_panel: pd.DataFrame,
    exit_treatments: dict[str, Any],
) -> pd.DataFrame:
    panel = stock_panel.copy()
    for ticker, treatment in exit_treatments.items():
        if ticker not in panel.columns:
            continue
        cash_start = treatment.get("cash_carry_start")
        if not cash_start:
            continue
        start = pd.Timestamp(cash_start)
        remove = pd.Timestamp(treatment["remove_effective_date"])
        mask = (panel.index >= start) & (panel.index < remove)
        existing = panel.loc[mask, ticker].notna()
        if existing.any():
            dates = panel.loc[mask].index[existing].strftime("%Y-%m-%d").tolist()[:5]
            raise ValueError(f"Cash-carry dates overlap market returns for {ticker}: {dates}")
        panel.loc[mask, ticker] = 0.0
    return panel


def active_universe_schedule(
    included: list[str],
    trading_dates: pd.DatetimeIndex,
    exit_treatments: dict[str, Any],
) -> dict[str, Any]:
    schedule = []
    for year in sorted(set(trading_dates.year)):
        year_dates = trading_dates[trading_dates.year == year]
        rebalance_date = year_dates.min()
        active = []
        for ticker in included:
            treatment = exit_treatments.get(ticker)
            if treatment is None or rebalance_date < pd.Timestamp(
                treatment["remove_effective_date"]
            ):
                active.append(ticker)
        schedule.append(
            {
                "year": int(year),
                "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
                "security_count": len(active),
                "active_tickers": active,
            }
        )
    return {
        "schema_version": 1,
        "rule": "Freeze the initial eligible universe and remove terminal securities at the next annual rebalance; introduce no later index additions.",
        "years": schedule,
    }


def audit_extreme_observations(
    daily: pd.DataFrame,
    review_config: dict[str, Any],
) -> dict[str, Any]:
    threshold = float(review_config.get("absolute_simple_return_threshold", 0.40))
    review_lookup = {
        (str(review["research_ticker"]), pd.Timestamp(review["date"])): review
        for review in review_config.get("reviews", [])
    }
    observed = daily.loc[
        daily["total_return"].abs().gt(threshold),
        ["research_ticker", "date", "total_return"],
    ]
    dispositions = []
    issues = []
    observed_keys = set()
    for row in observed.itertuples(index=False):
        key = (str(row.research_ticker), pd.Timestamp(row.date))
        observed_keys.add(key)
        review = review_lookup.get(key)
        if review is None:
            issues.append(f"{key[0]} {key[1].date()}: extreme return has no review")
            status = "missing_review"
            disposition = None
        else:
            status = str(review.get("status", ""))
            disposition = review.get("disposition")
            if status != "validated":
                issues.append(f"{key[0]} {key[1].date()}: review status is not validated")
            if disposition not in {"retain", "correct", "exclude"}:
                issues.append(f"{key[0]} {key[1].date()}: invalid or missing disposition")
        dispositions.append(
            {
                "research_ticker": key[0],
                "date": key[1].strftime("%Y-%m-%d"),
                "total_return": float(row.total_return),
                "status": status,
                "disposition": disposition,
                "review_id": review.get("review_id") if review else None,
                "source_url": review.get("source_url") if review else None,
                "rationale": review.get("rationale") if review else None,
            }
        )
    unused_reviews = sorted(
        f"{ticker} {date.strftime('%Y-%m-%d')}"
        for ticker, date in set(review_lookup) - observed_keys
    )
    issues.extend(
        f"Configured extreme review does not match an extreme row: {value}"
        for value in unused_reviews
    )
    return {
        "schema_version": int(review_config.get("schema_version", 0)),
        "absolute_simple_return_threshold": threshold,
        "observed_extreme_count": len(dispositions),
        "dispositions": dispositions,
        "unused_reviews": unused_reviews,
        "issues": issues,
    }


def numeric_integrity_audit(daily: pd.DataFrame) -> dict[str, int]:
    observed = daily.loc[daily["observation_status"].ne("unexplained_missing")].copy()
    required = ["open", "high", "low", "close", "volume", "transactions"]
    values = observed[required].apply(pd.to_numeric, errors="coerce")
    nonfinite_rows = int((~np.isfinite(values.to_numpy(dtype=float))).any(axis=1).sum())
    ohlc = values[["open", "high", "low", "close"]]
    ohlc_violations = int(
        (
            ohlc["high"].lt(ohlc[["open", "close", "low"]].max(axis=1))
            | ohlc["low"].gt(ohlc[["open", "close", "high"]].min(axis=1))
            | ohlc["high"].lt(ohlc["low"])
        ).sum()
    )
    return {
        "nonfinite_required_numeric_rows": nonfinite_rows,
        "ohlc_ordering_violations": ohlc_violations,
        "negative_volume_rows": int(values["volume"].lt(0).sum()),
        "negative_transaction_rows": int(values["transactions"].lt(0).sum()),
    }


def quality_report(
    daily: pd.DataFrame,
    coverage: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    manifest: dict[str, Any],
    event_reviews: dict[str, Any],
    expected_manual_action_ids: set[str],
    exit_audit: dict[str, Any],
    minimum_coverage: float,
    lifecycle_audit: dict[str, Any] | None = None,
    reference_reconciliation: list[dict[str, Any]] | None = None,
    observation_review_config: dict[str, Any] | None = None,
    universe_provenance_status: str = "unknown",
    input_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    lifecycle_audit = lifecycle_audit or {"issues": [], "classification_counts": {}}
    reference_reconciliation = reference_reconciliation or []
    observation_review_config = observation_review_config or {
        "schema_version": 0,
        "absolute_simple_return_threshold": 0.40,
        "reviews": [],
    }
    duplicates = int(daily.duplicated(["date", "research_ticker"]).sum())
    flags: defaultdict[str, int] = defaultdict(int)
    for value in daily["quality_flag"]:
        for flag in filter(None, str(value).split(";")):
            flags[flag] += 1
    applied_manual_action_ids = set(daily["manual_action_id"].dropna().astype(str))
    missing_manual_action_ids = sorted(expected_manual_action_ids - applied_manual_action_ids)
    unresolved_event_reviews = sorted(
        ticker
        for ticker, review in event_reviews.items()
        if not str(review.get("status", "")).startswith("resolved")
    )
    numeric_audit = numeric_integrity_audit(daily)
    extreme_audit = audit_extreme_observations(daily, observation_review_config)
    reference_counts: defaultdict[str, int] = defaultdict(int)
    for record in reference_reconciliation:
        reference_counts[str(record.get("disposition", "unclassified"))] += 1
    unclassified_reference_events = int(reference_counts.get("unclassified", 0))
    unmatched_reference_events = int(reference_counts.get("unmatched_price", 0))
    gate_status = "pass"
    if (
        duplicates
        or int(daily["close"].le(0).sum())
        or missing_manual_action_ids
        or exit_audit["issues"]
        or lifecycle_audit.get("issues")
        or unmatched_reference_events
        or unclassified_reference_events
        or extreme_audit["issues"]
        or any(numeric_audit.values())
    ):
        gate_status = "fail"
    elif unresolved_event_reviews:
        gate_status = "conditional_pass"
    return {
        "schema_version": 2,
        "gate_name": "data_construction_v2",
        "universe_provenance_status": universe_provenance_status,
        "output_scope": (
            "confirmed_foundation_input"
            if universe_provenance_status == "pass"
            else "provisional_pending_universe_provenance"
        ),
        "input_hashes": dict(sorted((input_hashes or {}).items())),
        "sample_start": manifest["sample_start"],
        "sample_end": manifest["sample_end"],
        "daily_provider_files": int(manifest["daily_file_count"]),
        "market_trading_dates": len(trading_dates),
        "candidate_securities": len(coverage),
        "included_securities": int(coverage["included"].sum()),
        "excluded_securities": int((~coverage["included"]).sum()),
        "minimum_initial_coverage": minimum_coverage,
        "duplicate_security_dates": duplicates,
        "nonpositive_closes": int(daily["close"].le(0).sum()),
        "missing_log_returns": int(daily["log_total_return"].isna().sum()),
        "split_event_security_dates": int(daily["split_factor"].ne(1).sum()),
        "dividend_event_security_dates": int(daily["cash_dividend"].ne(0).sum()),
        "quality_flag_counts": dict(sorted(flags.items())),
        "included_tickers": coverage.loc[coverage["included"], "ticker"].tolist(),
        "excluded": coverage.loc[
            ~coverage["included"],
            ["ticker", "company_name", "price_coverage", "return_coverage", "exclusion_reason"],
        ].to_dict("records"),
        "coverage": coverage.to_dict("records"),
        "expected_manual_action_ids": sorted(expected_manual_action_ids),
        "applied_manual_action_ids": sorted(applied_manual_action_ids),
        "missing_manual_action_ids": missing_manual_action_ids,
        "event_reviews": event_reviews,
        "unresolved_event_reviews": unresolved_event_reviews,
        "exit_treatment_audit": exit_audit,
        "lifecycle_audit": lifecycle_audit,
        "numeric_integrity_audit": numeric_audit,
        "reference_event_record_count": len(reference_reconciliation),
        "reference_event_detail_scope": "local_untracked_audit_only",
        "reference_event_detail_sha256": hashlib.sha256(
            json.dumps(
                reference_reconciliation,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "reference_event_disposition_counts": dict(sorted(reference_counts.items())),
        "unmatched_reference_events": unmatched_reference_events,
        "unclassified_reference_events": unclassified_reference_events,
        "extreme_observation_audit": extreme_audit,
        "gate_status": gate_status,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# FE5110 Data Quality Report",
        "",
        "## Data gate",
        "",
        f"**Status: {report['gate_status'].upper()}**",
        "",
        f"- Gate: {report['gate_name']}",
        f"- Output scope: {report['output_scope']}",
        f"- Universe provenance: {report['universe_provenance_status']}",
        f"- Provider daily files: {report['daily_provider_files']}",
        f"- Market trading dates: {report['market_trading_dates']}",
        f"- Candidate securities: {report['candidate_securities']}",
        f"- Included after the initial-window coverage screen: {report['included_securities']}",
        f"- Excluded: {report['excluded_securities']}",
        f"- Duplicate security-date rows: {report['duplicate_security_dates']}",
        f"- Non-positive closes: {report['nonpositive_closes']}",
        f"- Unexplained lifecycle issues: {len(report['lifecycle_audit']['issues'])}",
        f"- Unmatched reference events: {report['unmatched_reference_events']}",
        f"- Extreme-review issues: {len(report['extreme_observation_audit']['issues'])}",
        "",
        "## Exclusions",
        "",
        "| Ticker | Company | Price coverage | Return coverage | Reason |",
        "|---|---|---:|---:|---|",
    ]
    for row in report["excluded"]:
        lines.append(
            f"| {row['ticker']} | {row['company_name']} | {row['price_coverage']:.3%} | "
            f"{row['return_coverage']:.3%} | {row['exclusion_reason']} |"
        )
    if not report["excluded"]:
        lines.append("| — | None | — | — | — |")
    lines.extend(
        [
            "",
            "## Quality flags",
            "",
            "| Flag | Count |",
            "|---|---:|",
        ]
    )
    for flag, count in report["quality_flag_counts"].items():
        lines.append(f"| {flag} | {count} |")
    if not report["quality_flag_counts"]:
        lines.append("| None | 0 |")
    lines.extend(
        [
            "",
            "## Corporate-action reviews still requiring validation",
            "",
        ]
    )
    if report["unresolved_event_reviews"]:
        for ticker in report["unresolved_event_reviews"]:
            review = report["event_reviews"][ticker]
            lines.append(f"- **{ticker} ({review['status']}):** {review['note']}")
    else:
        lines.append("- None. All prespecified complex events have a documented treatment.")
    lines.extend(
        [
            "",
            "## Manual corporate-action reconciliations",
            "",
            f"- Expected actions: {len(report['expected_manual_action_ids'])}",
            f"- Applied actions: {len(report['applied_manual_action_ids'])}",
            f"- Missing actions: {len(report['missing_manual_action_ids'])}",
            "",
            "## Terminal-series audit",
            "",
            f"- Early-ending series: {len(report['exit_treatment_audit']['early_endings'])}",
            f"- Documented exit treatments: {len(report['exit_treatment_audit']['documented_exit_tickers'])}",
            f"- Exit-treatment issues: {len(report['exit_treatment_audit']['issues'])}",
        ]
    )
    lines.extend(
        [
            "",
            "## Lifecycle completion",
            "",
            f"- Verified halts applied: {report['lifecycle_audit'].get('verified_halts_applied', 0)}",
            f"- Unexplained missing active dates: "
            f"{report['lifecycle_audit'].get('classification_counts', {}).get('unexplained_missing', 0)}",
            f"- Terminal-cash security dates: "
            f"{report['lifecycle_audit'].get('classification_counts', {}).get('terminal_cash', 0)}",
            "",
            "## Reference-event reconciliation",
            "",
        ]
    )
    for disposition, count in report["reference_event_disposition_counts"].items():
        lines.append(f"- {disposition}: {count}")
    if not report["reference_event_disposition_counts"]:
        lines.append("- No provider events were supplied.")
    lines.extend(
        [
            "",
            "## Extreme observations",
            "",
            f"- Reviewed extremes: {report['extreme_observation_audit']['observed_extreme_count']}",
            f"- Review issues: {len(report['extreme_observation_audit']['issues'])}",
            "",
            "The coverage decision uses only the configured 2017–2019 initial training window. "
            "Later availability is not used to select the frozen universe.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/data_config.toml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)
    study = config["study"]
    paths = config["paths"]
    universe = load_json(project_path(project_root, paths["universe_json"]))
    master = load_json(project_path(project_root, paths["security_master_json"]))
    manual_action_config = load_json(
        project_path(project_root, paths["manual_corporate_actions_json"])
    )
    lifecycle_config = optional_json(
        project_path(
            project_root,
            paths.get("lifecycle_events_json", "config/lifecycle_events.json"),
        ),
        {"schema_version": 2, "securities": {}, "verified_halts": []},
    )
    observation_review_config = optional_json(
        project_path(
            project_root,
            paths.get("observation_reviews_json", "config/observation_reviews.json"),
        ),
        {"schema_version": 2, "absolute_simple_return_threshold": 0.40, "reviews": []},
    )
    market_manifest_path = project_path(project_root, paths["market_input_manifest_json"])
    manifest = load_json(market_manifest_path)
    if manifest.get("verification_status") != "pass":
        raise ValueError("Independent market-input manifest has not passed verification")
    if (
        manifest["sample_start"] != study["sample_start"]
        or manifest["sample_end"] != study["sample_end"]
    ):
        raise ValueError("Download manifest and configured sample dates differ")

    segments = build_segments(
        universe["constituents"], master, study["sample_start"], study["sample_end"]
    )
    prices, ancillary_prices = read_daily_files(
        project_root, manifest, segments, ancillary_tickers(manual_action_config)
    )
    prices = add_synthetic_primary_rows(prices, manual_action_config)
    reference_dir = project_path(project_root, paths["corporate_actions_directory"])
    verify_reference_files_against_manifest(project_root, manifest, reference_dir)
    split_daily, dividend_daily, reference_records = aggregate_actions_v2(
        reference_dir, segments, manual_action_config
    )
    manual_actions = manual_action_table(manual_action_config, ancillary_prices)
    trading_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime([row["date"] for row in manifest["daily_files"]]).unique())
    )
    corporate_action_keys = {
        (str(row.research_ticker), pd.Timestamp(row.date))
        for action_frame in (split_daily, dividend_daily, manual_actions)
        for row in action_frame[["research_ticker", "date"]].itertuples(index=False)
    }
    prices, lifecycle_audit = complete_lifecycle_grid(
        prices,
        trading_dates,
        segments,
        lifecycle_config,
        master.get("exit_treatments", {}),
        corporate_action_keys,
    )
    daily = construct_returns(prices, split_daily, dividend_daily, manual_actions)
    reference_reconciliation = reconcile_reference_events(reference_records, prices)
    coverage = coverage_table(
        daily,
        universe["constituents"],
        trading_dates,
        pd.Timestamp(study["initial_training_start"]),
        pd.Timestamp(study["initial_training_end"]),
        float(study["minimum_initial_coverage"]),
    )
    included = coverage.loc[coverage["included"], "ticker"].tolist()
    panel = (
        daily.loc[daily["research_ticker"].isin(included)]
        .pivot(index="date", columns="research_ticker", values="log_total_return")
        .reindex(trading_dates)
        .sort_index()
    )
    panel.index.name = "date"
    simple_panel = (
        daily.loc[daily["research_ticker"].isin(included)]
        .pivot(index="date", columns="research_ticker", values="total_return")
        .reindex(trading_dates)
        .sort_index()
    )
    simple_panel.index.name = "date"
    exit_treatments = master.get("exit_treatments", {})
    exit_audit = audit_exit_treatments(
        daily, included, pd.Timestamp(study["sample_end"]), exit_treatments
    )
    portfolio_panel = build_portfolio_panel(panel, exit_treatments)
    simple_portfolio_panel = build_portfolio_panel(simple_panel, exit_treatments)
    active_schedule = active_universe_schedule(included, trading_dates, exit_treatments)

    security_daily_path = project_path(project_root, paths["security_daily_parquet"])
    panel_path = project_path(project_root, paths["return_panel_parquet"])
    portfolio_panel_path = project_path(project_root, paths["portfolio_return_panel_parquet"])
    simple_panel_path = (
        project_path(project_root, paths["simple_return_panel_parquet"])
        if paths.get("simple_return_panel_parquet")
        else panel_path.with_name("daily_simple_total_returns.parquet")
    )
    simple_portfolio_panel_path = (
        project_path(project_root, paths["portfolio_simple_return_panel_parquet"])
        if paths.get("portfolio_simple_return_panel_parquet")
        else portfolio_panel_path.with_name("portfolio_constituent_simple_returns.parquet")
    )
    final_universe_path = project_path(project_root, paths["final_universe_json"])
    active_universe_path = project_path(project_root, paths["active_universe_json"])
    quality_json_path = project_path(project_root, paths["data_quality_json"])
    quality_local_json_path = project_path(
        project_root,
        paths.get("data_quality_local_json", "data/audit/data_quality_report.local.json"),
    )
    quality_md_path = project_path(project_root, paths["data_quality_markdown"])
    for path in (
        security_daily_path,
        panel_path,
        portfolio_panel_path,
        simple_panel_path,
        simple_portfolio_panel_path,
        final_universe_path,
        active_universe_path,
        quality_json_path,
        quality_local_json_path,
        quality_md_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    write_parquet_atomic(security_daily_path, daily, index=False, compression="zstd")
    write_parquet_atomic(panel_path, panel, index=True, compression="zstd")
    write_parquet_atomic(portfolio_panel_path, portfolio_panel, index=True, compression="zstd")
    write_parquet_atomic(simple_panel_path, simple_panel, index=True, compression="zstd")
    write_parquet_atomic(
        simple_portfolio_panel_path,
        simple_portfolio_panel,
        index=True,
        compression="zstd",
    )
    frozen_constituents = []
    coverage_lookup = coverage.set_index("ticker").to_dict("index")
    for row in universe["constituents"]:
        metrics = coverage_lookup[row["ticker"]]
        frozen_constituents.append(
            {
                **row,
                "initial_universe_status": "included" if metrics["included"] else "excluded",
                "initial_price_coverage": metrics["price_coverage"],
                "initial_return_coverage": metrics["return_coverage"],
                "exclusion_reason": metrics["exclusion_reason"],
            }
        )
    final_universe = {
        **{key: value for key, value in universe.items() if key != "constituents"},
        "schema_version": 2,
        "status": (
            "frozen"
            if universe.get("provenance_status") == "pass"
            else "provisional_pending_universe_provenance"
        ),
        "coverage_screen_window": [study["initial_training_start"], study["initial_training_end"]],
        "minimum_initial_coverage": float(study["minimum_initial_coverage"]),
        "candidate_security_count": len(universe["constituents"]),
        "included_security_count": len(included),
        "security_count": len(included),
        "included_tickers": included,
        "constituents": frozen_constituents,
    }
    write_json_atomic(final_universe_path, final_universe)
    write_json_atomic(active_universe_path, active_schedule)
    report = quality_report(
        daily,
        coverage,
        trading_dates,
        manifest,
        master.get("event_reviews", {}),
        {action["action_id"] for action in manual_action_config.get("actions", [])},
        exit_audit,
        float(study["minimum_initial_coverage"]),
        lifecycle_audit,
        reference_reconciliation,
        observation_review_config,
        str(universe.get("provenance_status", "unknown")),
        {
            "data_config_sha256": sha256_file(config_path),
            "lifecycle_events_sha256": sha256_file(
                project_path(
                    project_root,
                    paths.get("lifecycle_events_json", "config/lifecycle_events.json"),
                )
            ),
            "manual_corporate_actions_sha256": sha256_file(
                project_path(project_root, paths["manual_corporate_actions_json"])
            ),
            "market_input_manifest_sha256": sha256_file(market_manifest_path),
            "observation_reviews_sha256": sha256_file(
                project_path(
                    project_root,
                    paths.get("observation_reviews_json", "config/observation_reviews.json"),
                )
            ),
            "security_master_sha256": sha256_file(
                project_path(project_root, paths["security_master_json"])
            ),
            "universe_json_sha256": sha256_file(project_path(project_root, paths["universe_json"])),
            "universe_source_manifest_sha256": sha256_file(
                project_path(project_root, paths["universe_source_manifest_json"])
            ),
        },
    )
    report["output_hashes"] = {
        "active_universe_sha256": sha256_file(active_universe_path),
        "final_universe_sha256": sha256_file(final_universe_path),
        "portfolio_simple_return_panel_sha256": sha256_file(simple_portfolio_panel_path),
    }
    local_report = {**report, "reference_event_reconciliation": reference_reconciliation}
    write_json_atomic(quality_local_json_path, local_report)
    write_json_atomic(quality_json_path, report)
    write_text_atomic(quality_md_path, markdown_report(report))
    print(f"Security-day data: {security_daily_path}")
    print(f"Return panel: {panel_path} ({panel.shape[0]} dates x {panel.shape[1]} securities)")
    print(f"Simple-return panel: {simple_panel_path}")
    print(f"Portfolio constituent panel: {portfolio_panel_path}")
    print(f"Simple portfolio constituent panel: {simple_portfolio_panel_path}")
    print(f"Frozen universe: {final_universe_path} ({len(included)} included)")
    print(f"Active-universe schedule: {active_universe_path}")
    print(f"Data gate: {report['gate_status'].upper()}")
    if report["gate_status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
