#!/usr/bin/env python3
"""Build a clearly provisional Yahoo Finance universe-reference table.

Yahoo Finance does not provide the licensed, point-in-time S&P 100/GICS
corroboration required by ``foundation_v2``.  This script therefore writes only
to provisional/ignored paths and never populates the licensed-universe gate
input.  It combines current Yahoo search metadata with a check for a price on
2 January 2020 and retains the frozen candidate fields for comparison.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


SEARCH_ENDPOINT = "https://query2.finance.yahoo.com/v1/finance/search"
CHART_ENDPOINT = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_TERMS_URL = "https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html"
TARGET_DATE = dt.date(2020, 1, 2)
PERIOD1 = int(dt.datetime(2020, 1, 1, tzinfo=dt.UTC).timestamp())
PERIOD2 = int(dt.datetime(2020, 1, 5, tzinfo=dt.UTC).timestamp())

# Yahoo uses a hyphen for Berkshire Hathaway's share-class ticker. Facebook's
# Yahoo series continues under META; the alias is disclosed on every FB row.
YAHOO_SYMBOL_ALIASES = {
    "BK": "BNY",
    "BRK.B": "BRK-B",
    "FB": "META",
    "UTX": "RTX",
}

OUTPUT_COLUMNS = (
    "reference_ticker",
    "reference_company_name",
    "reference_gics_sector_2020",
    "reference_gics_sub_industry_2020",
    "reference_as_of_date",
    "yahoo_query_symbol",
    "yahoo_symbol",
    "yahoo_short_name_current",
    "yahoo_long_name_current",
    "yahoo_exchange_current",
    "yahoo_quote_type_current",
    "yahoo_sector_current",
    "yahoo_industry_current",
    "yahoo_price_date",
    "yahoo_close_2020_01_02",
    "yahoo_adjusted_close_2020_01_02",
    "yahoo_currency",
    "yahoo_chart_exchange",
    "yahoo_first_trade_date",
    "metadata_status",
    "price_status",
    "provenance_status",
    "limitations",
    "search_source_url",
    "chart_source_url",
)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_candidate(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("constituents")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Candidate universe has no constituents: {path}")
    tickers = [str(row.get("ticker", "")).strip() for row in rows]
    if any(not ticker for ticker in tickers) or len(tickers) != len(set(tickers)):
        raise ValueError("Candidate universe tickers must be non-empty and unique")
    return sorted(rows, key=lambda row: row["ticker"])


def source_url(endpoint: str, params: dict[str, str | int]) -> str:
    return endpoint + "?" + urllib.parse.urlencode(params)


def fetch_json(url: str, retries: int = 4) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; FE5110 academic research; contact local user)",
            "Accept": "application/json",
        },
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(f"Yahoo request failed after {retries} attempts: {url}") from exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def select_quote(payload: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    quotes = payload.get("quotes") or []
    exact = [row for row in quotes if str(row.get("symbol", "")).upper() == symbol.upper()]
    return exact[0] if exact else None


def chart_observation(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    results = ((payload.get("chart") or {}).get("result") or [])
    if not results:
        return {}, None
    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = (((result.get("indicators") or {}).get("quote") or [{}])[0])
    adjusted = (((result.get("indicators") or {}).get("adjclose") or [{}])[0])
    closes = quote.get("close") or []
    adjusted_closes = adjusted.get("adjclose") or []
    for index, timestamp in enumerate(timestamps):
        date = dt.datetime.fromtimestamp(timestamp, tz=dt.UTC).date()
        if date == TARGET_DATE:
            return result.get("meta") or {}, {
                "date": date.isoformat(),
                "close": closes[index] if index < len(closes) else None,
                "adjusted_close": (
                    adjusted_closes[index] if index < len(adjusted_closes) else None
                ),
            }
    return result.get("meta") or {}, None


def raw_path(root: Path, ticker: str, kind: str) -> Path:
    safe_ticker = ticker.replace(".", "_").replace("-", "_")
    return root / f"{safe_ticker}_{kind}.json"


def write_raw(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload), encoding="utf-8")


def cached_or_fetched(
    path: Path, url: str, refresh: bool
) -> tuple[dict[str, Any], bool]:
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8")), False
    payload = fetch_json(url)
    write_raw(path, payload)
    return payload, True


def build_reference(
    candidate_path: Path,
    output_path: Path,
    manifest_path: Path,
    raw_root: Path,
    delay_seconds: float,
    refresh: bool = False,
) -> dict[str, Any]:
    candidates = read_candidate(candidate_path)
    retrieved_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for candidate in candidates:
        ticker = candidate["ticker"]
        yahoo_symbol = YAHOO_SYMBOL_ALIASES.get(ticker, ticker)
        search_params = {"q": yahoo_symbol, "quotesCount": 5, "newsCount": 0}
        chart_params = {
            "period1": PERIOD1,
            "period2": PERIOD2,
            "interval": "1d",
            "events": "history",
        }
        search_url = source_url(SEARCH_ENDPOINT, search_params)
        chart_url = source_url(CHART_ENDPOINT.format(symbol=yahoo_symbol), chart_params)

        search_payload: dict[str, Any] = {}
        chart_payload: dict[str, Any] = {}
        search_error = ""
        chart_error = ""
        search_fetched = False
        chart_fetched = False
        try:
            search_payload, search_fetched = cached_or_fetched(
                raw_path(raw_root, ticker, "search"), search_url, refresh
            )
        except RuntimeError as exc:
            search_error = str(exc)
        if search_fetched:
            time.sleep(delay_seconds)
        try:
            chart_payload, chart_fetched = cached_or_fetched(
                raw_path(raw_root, ticker, "chart"), chart_url, refresh
            )
        except RuntimeError as exc:
            chart_error = str(exc)
        if chart_fetched:
            time.sleep(delay_seconds)

        quote = select_quote(search_payload, yahoo_symbol)
        if quote is None:
            fallback_params = {
                "q": f"{yahoo_symbol} {candidate['company_name']}",
                "quotesCount": 10,
                "newsCount": 0,
            }
            fallback_url = source_url(SEARCH_ENDPOINT, fallback_params)
            try:
                fallback_payload, fallback_fetched = cached_or_fetched(
                    raw_path(raw_root, ticker, "search_fallback"), fallback_url, refresh
                )
                quote = select_quote(fallback_payload, yahoo_symbol)
                if fallback_fetched:
                    time.sleep(delay_seconds)
            except RuntimeError as exc:
                if not search_error:
                    search_error = str(exc)
        chart_meta, observation = chart_observation(chart_payload)
        if search_error or chart_error:
            failures.append(
                {"ticker": ticker, "search_error": search_error, "chart_error": chart_error}
            )
        limitations = [
            "Yahoo metadata is current, not point-in-time as of 2020-01-02",
            "Yahoo sector/industry labels are not accepted as point-in-time GICS corroboration",
            "Yahoo ticker is not a permanent security or issuer identifier",
        ]
        if ticker in YAHOO_SYMBOL_ALIASES:
            limitations.append(f"Yahoo symbol alias applied: {ticker}->{yahoo_symbol}")
        rows.append(
            {
                "reference_ticker": ticker,
                "reference_company_name": candidate["company_name"],
                "reference_gics_sector_2020": candidate["gics_sector"],
                "reference_gics_sub_industry_2020": candidate["gics_sub_industry"],
                "reference_as_of_date": candidate["as_of_date"],
                "yahoo_query_symbol": yahoo_symbol,
                "yahoo_symbol": (quote or {}).get("symbol") or chart_meta.get("symbol") or "",
                "yahoo_short_name_current": (quote or {}).get("shortname") or chart_meta.get("shortName") or "",
                "yahoo_long_name_current": (quote or {}).get("longname") or chart_meta.get("longName") or "",
                "yahoo_exchange_current": (quote or {}).get("exchDisp") or (quote or {}).get("exchange") or "",
                "yahoo_quote_type_current": (quote or {}).get("quoteType") or chart_meta.get("instrumentType") or "",
                "yahoo_sector_current": (quote or {}).get("sector") or "",
                "yahoo_industry_current": (quote or {}).get("industry") or "",
                "yahoo_price_date": (observation or {}).get("date") or "",
                "yahoo_close_2020_01_02": (observation or {}).get("close"),
                "yahoo_adjusted_close_2020_01_02": (observation or {}).get("adjusted_close"),
                "yahoo_currency": chart_meta.get("currency") or "",
                "yahoo_chart_exchange": chart_meta.get("fullExchangeName") or chart_meta.get("exchangeName") or "",
                "yahoo_first_trade_date": (
                    dt.datetime.fromtimestamp(chart_meta["firstTradeDate"], tz=dt.UTC).date().isoformat()
                    if chart_meta.get("firstTradeDate")
                    else ""
                ),
                "metadata_status": (
                    "found_current_search"
                    if quote
                    else "found_chart_only"
                    if chart_meta
                    else "not_found_or_inactive"
                ),
                "price_status": "found_2020_01_02" if observation else "not_found_2020_01_02",
                "provenance_status": "provisional_non_licensed",
                "limitations": "; ".join(limitations),
                "search_source_url": search_url,
                "chart_source_url": chart_url,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": 1,
        "dataset": "Yahoo Finance provisional universe reference",
        "status": "provisional_non_licensed",
        "retrieved_at_utc": retrieved_at,
        "target_date": TARGET_DATE.isoformat(),
        "candidate_path": candidate_path.as_posix(),
        "candidate_sha256": sha256_file(candidate_path),
        "output_path": output_path.as_posix(),
        "output_sha256": sha256_file(output_path),
        "row_count": len(rows),
        "metadata_found_count": sum(
            row["metadata_status"] != "not_found_or_inactive" for row in rows
        ),
        "price_found_count": sum(row["price_status"] == "found_2020_01_02" for row in rows),
        "request_failure_count": len(failures),
        "request_failures": failures,
        "source": {
            "name": "Yahoo Finance",
            "search_endpoint": SEARCH_ENDPOINT,
            "chart_endpoint": CHART_ENDPOINT,
            "terms_url": YAHOO_TERMS_URL,
        },
        "gate_eligibility": {
            "foundation_v2_universe_provenance": False,
            "reason": (
                "Yahoo does not supply the authorized point-in-time S&P 100 membership, "
                "historical GICS, and permanent issuer/security IDs required by the frozen protocol."
            ),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-json",
        type=Path,
        default=Path("data/raw/universe_sp100_2020-01-02.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/provisional/yahoo_universe_reference.csv"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/manifests/yahoo_universe_reference.local.json"),
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/raw/yahoo/universe_reference")
    )
    parser.add_argument("--delay-seconds", type=float, default=0.20)
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached raw Yahoo responses"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_reference(
        args.candidate_json,
        args.output,
        args.manifest_output,
        args.raw_root,
        args.delay_seconds,
        args.refresh,
    )
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
