#!/usr/bin/env python3
"""Build and provenance-check the fixed S&P 100 research universe.

The Wikipedia revisions are reproducible secondary-source inputs. They are
never sufficient for the final provenance gate: an authorised point-in-time
extract using ``LICENSED_COLUMNS`` must also reconcile without differences.
No licensed payload is copied into the public source manifest.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any


GICS_SECTORS = (
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
)

EXPECTED_SP100_REVISION = 929329274
EXPECTED_SP500_REVISION = 933762580
EXPECTED_SP100_TITLE = "S&P 100"
EXPECTED_SP500_TITLE = "List of S&P 500 companies"
EXPECTED_SP100_TIMESTAMP = "2019-12-05T03:03:35Z"
EXPECTED_SP500_TIMESTAMP = "2020-01-02T22:21:11Z"
FROZEN_AS_OF_DATE = "2020-01-02"
FROZEN_MEMBERSHIP_TABLE_DATE = "2019-11-21"
FROZEN_SECURITY_COUNT = 101
FROZEN_SECTOR_COUNT = 11

LICENSED_COLUMNS = (
    "security_id",
    "issuer_id",
    "ticker",
    "share_class",
    "company_name",
    "gics_sector",
    "gics_sub_industry",
    "membership_date",
    "as_of_date",
    "source_record_id",
)

KNOWN_EVENT_REVIEWS = {
    "AGN": "Acquired by AbbVie in 2020; verify terminal return and removal date.",
    "FB": "Ticker changed to META in 2022; join the two ticker segments by security identifier.",
    "RTN": "Merged with United Technologies in 2020; verify terminal consideration and removal date.",
    "UTX": "Renamed RTX after the Raytheon merger; continue the series using point-in-time identifiers.",
}

WIKIPEDIA_LICENSE = {
    "name": "Creative Commons Attribution-ShareAlike 4.0 International",
    "identifier": "CC-BY-SA-4.0",
    "terms_url": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
    "access_class": "public_secondary_source",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def revision_record(
    path: Path, project_root: Path | None = None
) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        page = payload["query"]["pages"][0]
        revision = page["revisions"][0]
        content = revision["slots"]["main"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Malformed MediaWiki revision payload: {path}") from exc
    resolved = path.resolve()
    if project_root is not None:
        try:
            payload_path = resolved.relative_to(project_root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(f"Revision payload must be archived inside the project: {path}") from exc
    else:
        payload_path = path.as_posix()
    return {
        "page_title": page["title"],
        "revision_id": int(revision["revid"]),
        "revision_timestamp": revision["timestamp"],
        "payload_path": payload_path,
        "payload_bytes": path.stat().st_size,
        "payload_sha256": sha256_file(path),
        "license": WIKIPEDIA_LICENSE,
    }, content


def clean_wikitext(value: str) -> str:
    value = re.sub(r"<ref.*?</ref>", "", value, flags=re.DOTALL)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"\[\[[^]|]+\|([^]]+)\]\]", r"\1", value)
    value = re.sub(r"\[\[([^]]+)\]\]", r"\1", value)
    value = re.sub(r"\{\{[^{}]*\}\}", "", value)
    value = value.replace("''", "")
    return html.unescape(re.sub(r"\s+", " ", value).strip())


def parse_sp100(content: str) -> list[dict[str, str]]:
    match = re.search(
        r"==\s*Components\s*==(.*?)==\s*Statistics\s*==",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        raise ValueError("S&P 100 components table not found")

    records = []
    for chunk in re.split(r"\n\|-\s*\n", match.group(1)):
        values = [line[2:].strip() for line in chunk.splitlines() if line.startswith("| ")]
        if len(values) >= 2 and re.fullmatch(r"[A-Z][A-Z.]*", values[0]):
            records.append({"ticker": values[0], "company_name": clean_wikitext(values[1])})
    return records


def parse_sp500_gics(content: str) -> dict[str, dict[str, str]]:
    match = re.search(
        r"==\s*S&P 500 component stocks\s*==(.*?)==\s*Selected changes",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        raise ValueError("S&P 500 component table not found")

    sector_pattern = "|".join(re.escape(item) for item in GICS_SECTORS)
    records = {}
    for chunk in re.split(r"\n\|-\s*\n", match.group(1)):
        ticker_match = re.search(r"\{\{[^{}|]*[Ss]ymbol\|([^}|]+)", chunk)
        if not ticker_match:
            continue
        gics_match = re.search(
            rf"\|\|\s*(?P<sector>{sector_pattern})\s*(?:\|\||\n\|)\s*(?P<industry>[^|\n]+)",
            chunk,
        )
        if not gics_match:
            continue
        records[ticker_match.group(1).strip()] = {
            "gics_sector": clean_wikitext(gics_match.group("sector")),
            "gics_sub_industry": clean_wikitext(gics_match.group("industry")),
        }
    return records


def research_identifiers(ticker: str) -> tuple[str, str, str]:
    issuer = "ALPHABET" if ticker in {"GOOG", "GOOGL"} else ticker.split(".", maxsplit=1)[0]
    if ticker == "GOOG":
        share_class = "Class C"
    elif ticker == "GOOGL":
        share_class = "Class A"
    elif "." in ticker:
        share_class = ticker.split(".", maxsplit=1)[1]
    else:
        share_class = "common"
    return f"SP100-20200102:{ticker}", f"ISSUER:{issuer}", share_class


def build_secondary_records(
    membership: list[dict[str, str]],
    gics: dict[str, dict[str, str]],
    as_of_date: str,
    membership_table_date: str,
) -> tuple[list[dict[str, str]], list[str]]:
    unmatched = [row["ticker"] for row in membership if row["ticker"] not in gics]
    records: list[dict[str, str]] = []
    for row in membership:
        ticker = row["ticker"]
        if ticker not in gics:
            continue
        security_id, issuer_id, share_class = research_identifiers(ticker)
        records.append(
            {
                "security_id": security_id,
                "issuer_id": issuer_id,
                "ticker": ticker,
                "share_class": share_class,
                "company_name": row["company_name"],
                **gics[ticker],
                "membership_date": membership_table_date,
                "as_of_date": as_of_date,
                "source_record_id": f"wikipedia:{ticker}",
                # Backwards-compatible names consumed by the return builder.
                "membership_as_of": as_of_date,
                "membership_table_date": membership_table_date,
                "initial_universe_status": "candidate_pending_licensed_corroboration",
                "known_ticker_event_review": KNOWN_EVENT_REVIEWS.get(ticker, ""),
            }
        )
    return records, unmatched


def read_licensed_universe(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(LICENSED_COLUMNS) - set(reader.fieldnames or []))
        extra = sorted(set(reader.fieldnames or []) - set(LICENSED_COLUMNS))
        if missing or extra:
            raise ValueError(f"Licensed-universe columns differ: missing={missing}, extra={extra}")
        rows = [{field: (row[field] or "").strip() for field in LICENSED_COLUMNS} for row in reader]
    if not rows:
        raise ValueError("Licensed-universe extract is empty")
    for unique_field in ("security_id", "ticker", "source_record_id"):
        values = [row[unique_field] for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(
                f"Licensed-universe extract contains duplicate {unique_field} values"
            )
    required_nonempty = set(LICENSED_COLUMNS)
    for index, row in enumerate(rows, start=2):
        empty = sorted(field for field in required_nonempty if not row[field])
        if empty:
            raise ValueError(f"Licensed-universe row {index} has empty required fields: {empty}")
        try:
            membership_date = dt.date.fromisoformat(row["membership_date"])
            as_of_date = dt.date.fromisoformat(row["as_of_date"])
        except ValueError as exc:
            raise ValueError(
                f"Licensed-universe row {index} has a non-ISO membership/as-of date"
            ) from exc
        if membership_date > as_of_date:
            raise ValueError(
                f"Licensed-universe row {index} has membership_date after as_of_date"
            )
        if row["gics_sector"] not in GICS_SECTORS:
            raise ValueError(
                f"Licensed-universe row {index} has unsupported GICS sector: "
                f"{row['gics_sector']!r}"
            )
    return sorted(rows, key=lambda row: row["ticker"])


def reconcile_licensed_universe(
    secondary: list[dict[str, str]], licensed: list[dict[str, str]] | None, as_of_date: str
) -> dict[str, Any]:
    if licensed is None:
        return {
            "status": "blocked_authorized_extract_absent",
            "missing_from_licensed": sorted(row["ticker"] for row in secondary),
            "extra_in_licensed": [],
            "field_differences": [],
        }

    secondary_by_ticker = {row["ticker"]: row for row in secondary}
    licensed_by_ticker = {row["ticker"]: row for row in licensed}
    missing = sorted(set(secondary_by_ticker) - set(licensed_by_ticker))
    extra = sorted(set(licensed_by_ticker) - set(secondary_by_ticker))
    differences: list[dict[str, str]] = []
    for ticker in sorted(set(secondary_by_ticker) & set(licensed_by_ticker)):
        left, right = secondary_by_ticker[ticker], licensed_by_ticker[ticker]
        for field in ("gics_sector", "gics_sub_industry"):
            if left[field] != right[field]:
                differences.append(
                    {"ticker": ticker, "field": field, "secondary": left[field], "licensed": right[field]}
                )
        if right["as_of_date"] != as_of_date:
            differences.append(
                {
                    "ticker": ticker,
                    "field": "as_of_date",
                    "secondary": as_of_date,
                    "licensed": right["as_of_date"],
                }
            )
    status = "pass" if not missing and not extra and not differences else "blocked_unresolved_differences"
    return {
        "status": status,
        "missing_from_licensed": missing,
        "extra_in_licensed": extra,
        "field_differences": differences,
    }


def public_reconciliation_summary(reconciliation: dict[str, Any]) -> dict[str, Any]:
    """Summarize reconciliation without publishing licensed row-level values."""

    return {
        "status": reconciliation["status"],
        "missing_from_licensed_count": len(reconciliation.get("missing_from_licensed", [])),
        "extra_in_licensed_count": len(reconciliation.get("extra_in_licensed", [])),
        "field_difference_count": len(reconciliation.get("field_differences", [])),
        "details_sha256": hashlib.sha256(
            canonical_json(reconciliation).encode("utf-8")
        ).hexdigest(),
        "detail_scope": "local_untracked_only",
    }


def build_source_manifest(
    sp100_meta: dict[str, Any],
    sp500_meta: dict[str, Any],
    licensed_path: Path | None,
    licensed_source_name: str,
    licensed_license_reference: str,
    reconciliation: dict[str, Any],
    checks: dict[str, Any],
) -> dict[str, Any]:
    licensed_present = licensed_path is not None
    license_metadata_complete = bool(licensed_source_name and licensed_license_reference)
    licensed_status = reconciliation["status"]
    if licensed_present and not license_metadata_complete:
        licensed_status = "blocked_license_metadata_absent"
    return {
        "schema_version": 2,
        "manifest_type": "universe_source_manifest",
        "sources": [
            {
                **sp100_meta,
                "source_id": "wikipedia_sp100_membership",
                "purpose": "S&P 100 membership secondary snapshot",
                "url": (
                    "https://en.wikipedia.org/w/index.php?title=S%26P_100"
                    f"&oldid={sp100_meta['revision_id']}"
                ),
            },
            {
                **sp500_meta,
                "source_id": "wikipedia_sp500_gics",
                "purpose": "Contemporaneous GICS secondary snapshot",
                "url": (
                    "https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies"
                    f"&oldid={sp500_meta['revision_id']}"
                ),
            },
        ],
        "licensed_corroboration": {
            "status": licensed_status,
            "source_name": licensed_source_name or None,
            "license_reference_recorded": bool(licensed_license_reference),
            "license_reference_sha256": (
                hashlib.sha256(licensed_license_reference.encode("utf-8")).hexdigest()
                if licensed_license_reference
                else None
            ),
            "payload_path": licensed_path.name if licensed_path else None,
            "payload_sha256": sha256_file(licensed_path) if licensed_path else None,
            "payload_bytes": licensed_path.stat().st_size if licensed_path else None,
            "raw_payload_publishable": False,
            "reconciliation": public_reconciliation_summary(reconciliation),
        },
        "checks": checks,
        "source_quality_note": (
            "Wikipedia revisions are reproducible secondary sources. The final universe-provenance "
            "gate passes only after an authorised point-in-time extract reconciles without differences."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp100-json", type=Path, required=True)
    parser.add_argument("--sp500-json", type=Path, required=True)
    parser.add_argument("--licensed-universe-csv", type=Path)
    parser.add_argument("--licensed-source-name", default="")
    parser.add_argument("--licensed-license-reference", default="")
    parser.add_argument(
        "--licensed-normalized-output",
        type=Path,
        default=Path("data/interim/licensed/universe_sp100_2020-01-02.json"),
        help="Untracked normalized licensed records; never merged into the public candidate file",
    )
    parser.add_argument(
        "--licensed-reconciliation-output",
        type=Path,
        default=Path("data/interim/licensed/universe_reconciliation.json"),
        help="Untracked row-level reconciliation details",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--provenance-audit-output", type=Path)
    parser.add_argument("--as-of-date", default=FROZEN_AS_OF_DATE)
    parser.add_argument("--membership-table-date", default=FROZEN_MEMBERSHIP_TABLE_DATE)
    parser.add_argument("--expected-securities", type=int, default=FROZEN_SECURITY_COUNT)
    parser.add_argument("--expected-sectors", type=int, default=FROZEN_SECTOR_COUNT)
    parser.add_argument("--expected-sp100-revision", type=int, default=EXPECTED_SP100_REVISION)
    parser.add_argument("--expected-sp500-revision", type=int, default=EXPECTED_SP500_REVISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen_cli_values = {
        "as_of_date": (args.as_of_date, FROZEN_AS_OF_DATE),
        "membership_table_date": (
            args.membership_table_date,
            FROZEN_MEMBERSHIP_TABLE_DATE,
        ),
        "expected_securities": (args.expected_securities, FROZEN_SECURITY_COUNT),
        "expected_sectors": (args.expected_sectors, FROZEN_SECTOR_COUNT),
        "expected_sp100_revision": (args.expected_sp100_revision, EXPECTED_SP100_REVISION),
        "expected_sp500_revision": (args.expected_sp500_revision, EXPECTED_SP500_REVISION),
    }
    changed = {
        name: {"received": received, "frozen": frozen}
        for name, (received, frozen) in frozen_cli_values.items()
        if received != frozen
    }
    if changed:
        raise ValueError(f"Protocol-critical universe parameters cannot be overridden: {changed}")

    project_root = Path(__file__).resolve().parents[1]
    sp100_meta, sp100_content = revision_record(args.sp100_json, project_root)
    sp500_meta, sp500_content = revision_record(args.sp500_json, project_root)
    if sp100_meta["revision_id"] != args.expected_sp100_revision:
        raise ValueError(
            f"S&P 100 revision mismatch: expected {args.expected_sp100_revision}, "
            f"found {sp100_meta['revision_id']}"
        )
    if sp500_meta["revision_id"] != args.expected_sp500_revision:
        raise ValueError(
            f"S&P 500 revision mismatch: expected {args.expected_sp500_revision}, "
            f"found {sp500_meta['revision_id']}"
        )
    expected_source_identity = (
        (sp100_meta, EXPECTED_SP100_TITLE, EXPECTED_SP100_TIMESTAMP, "S&P 100"),
        (sp500_meta, EXPECTED_SP500_TITLE, EXPECTED_SP500_TIMESTAMP, "S&P 500"),
    )
    for metadata, expected_title, expected_timestamp, label in expected_source_identity:
        if (
            metadata["page_title"] != expected_title
            or metadata["revision_timestamp"] != expected_timestamp
        ):
            raise ValueError(
                f"{label} source identity differs: expected title/timestamp "
                f"{expected_title!r}/{expected_timestamp}, found "
                f"{metadata['page_title']!r}/{metadata['revision_timestamp']}"
            )
    membership = parse_sp100(sp100_content)
    gics = parse_sp500_gics(sp500_content)
    records, unmatched = build_secondary_records(
        membership, gics, args.as_of_date, args.membership_table_date
    )
    if unmatched:
        raise ValueError(f"Missing GICS mapping: {unmatched}")

    sectors = sorted({row["gics_sector"] for row in records})
    if len(records) != args.expected_securities:
        raise ValueError(f"Expected {args.expected_securities} securities, found {len(records)}")
    if len(sectors) != args.expected_sectors:
        raise ValueError(f"Expected {args.expected_sectors} sectors, found {len(sectors)}: {sectors}")

    licensed = read_licensed_universe(args.licensed_universe_csv) if args.licensed_universe_csv else None
    reconciliation = reconcile_licensed_universe(records, licensed, args.as_of_date)
    license_metadata_complete = bool(args.licensed_source_name and args.licensed_license_reference)
    provenance_pass = reconciliation["status"] == "pass" and license_metadata_complete
    provenance_status = "pass" if provenance_pass else "blocked"
    if licensed is None:
        blockers = ["authorized_point_in_time_universe_extract_absent"]
    elif not license_metadata_complete:
        blockers = ["licensed_source_or_license_metadata_absent"]
    elif reconciliation["status"] != "pass":
        blockers = ["unresolved_universe_reconciliation_differences"]
    else:
        blockers = []

    public_records = [dict(row) for row in records]
    if provenance_pass:
        for row in public_records:
            row["initial_universe_status"] = "candidate_pending_full_coverage_test"

    normalized_licensed_universe: dict[str, Any] | None = None
    if licensed is not None:
        secondary_by_ticker = {row["ticker"]: row for row in records}
        normalized_records = []
        for corroborated in licensed:
            row = secondary_by_ticker.get(corroborated["ticker"], {})
            normalized_records.append(
                {
                    **row,
                    **corroborated,
                    "membership_as_of": corroborated["as_of_date"],
                    "membership_table_date": corroborated["membership_date"],
                    "initial_universe_status": (
                        "candidate_pending_full_coverage_test"
                        if provenance_pass
                        else "provisional_not_reconciled"
                    ),
                }
            )
        normalized_licensed_universe = {
            "schema_version": 2,
            "universe_name": "Licensed normalized S&P 100 point-in-time extract",
            "as_of_date": args.as_of_date,
            "status": "reconciled" if provenance_pass else "provisional_not_reconciled",
            "source_name": args.licensed_source_name or None,
            "license_reference": args.licensed_license_reference or None,
            "raw_payload_publishable": False,
            "security_count": len(normalized_records),
            "constituents": normalized_records,
        }

    checks = {
        "unmatched_tickers": unmatched,
        "security_count": len(records),
        "gics_sector_count": len(sectors),
        "sectors": sectors,
    }
    source_manifest = build_source_manifest(
        sp100_meta,
        sp500_meta,
        args.licensed_universe_csv,
        args.licensed_source_name,
        args.licensed_license_reference,
        reconciliation,
        checks,
    )
    universe = {
        "schema_version": 2,
        "universe_name": "S&P 100 point-in-time research universe",
        "membership_as_of": args.as_of_date,
        "membership_table_date": args.membership_table_date,
        "security_count": len(public_records),
        "gics_sector_count": len(sectors),
        "provenance_status": provenance_status,
        "status": (
            "candidate_pending_full_coverage_test"
            if provenance_pass
            else "candidate_blocked_pending_licensed_corroboration"
        ),
        "constituents": public_records,
    }
    audit = {
        "schema_version": 2,
        "gate": "universe_provenance",
        "status": provenance_status,
        "blockers": blockers,
        "source_manifest_sha256": hashlib.sha256(canonical_json(source_manifest).encode()).hexdigest(),
        "public_universe_sha256": hashlib.sha256(canonical_json(universe).encode()).hexdigest(),
        "reconciliation": public_reconciliation_summary(reconciliation),
    }

    for path, payload in ((args.output, universe), (args.metadata_output, source_manifest)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(payload), encoding="utf-8")
    if args.provenance_audit_output:
        args.provenance_audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_audit_output.write_text(canonical_json(audit), encoding="utf-8")
    if normalized_licensed_universe is not None:
        args.licensed_normalized_output.parent.mkdir(parents=True, exist_ok=True)
        args.licensed_normalized_output.write_text(
            canonical_json(normalized_licensed_universe), encoding="utf-8"
        )
        args.licensed_reconciliation_output.parent.mkdir(parents=True, exist_ok=True)
        args.licensed_reconciliation_output.write_text(
            canonical_json(reconciliation), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
