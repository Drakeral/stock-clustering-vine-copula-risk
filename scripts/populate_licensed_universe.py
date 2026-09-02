#!/usr/bin/env python3
"""Create and finalize a local worksheet for licensed-universe corroboration.

The worksheet places the archived public candidate beside blank, editable
licensed-source fields.  Candidate values are references only and are never
copied into the licensed columns.  Finalization validates the strict licensed
schema and requires exact universe/GICS reconciliation before writing the raw
licensed CSV consumed by ``prepare_universe.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts.prepare_universe import (
        LICENSED_COLUMNS,
        read_licensed_universe,
        reconcile_licensed_universe,
    )
except ModuleNotFoundError:  # Direct execution: ``python scripts/<name>.py``.
    from prepare_universe import (  # type: ignore[no-redef]
        LICENSED_COLUMNS,
        read_licensed_universe,
        reconcile_licensed_universe,
    )


REFERENCE_COLUMNS = (
    "reference_ticker",
    "reference_company_name",
    "reference_gics_sector",
    "reference_gics_sub_industry",
    "reference_as_of_date",
)
ENTRY_COLUMNS = tuple(f"licensed_{column}" for column in LICENSED_COLUMNS)
WORKSHEET_COLUMNS = (*REFERENCE_COLUMNS, *ENTRY_COLUMNS, "entry_status", "review_notes")


def read_candidate(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("constituents")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Candidate universe has no constituents: {path}")
    tickers = [str(row.get("ticker", "")).strip() for row in rows]
    if any(not ticker for ticker in tickers) or len(tickers) != len(set(tickers)):
        raise ValueError("Candidate universe tickers must be non-empty and unique")
    return sorted(rows, key=lambda row: row["ticker"])


def create_worksheet(candidate_path: Path, output_path: Path) -> int:
    rows = read_candidate(candidate_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WORKSHEET_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "reference_ticker": row["ticker"],
                    "reference_company_name": row["company_name"],
                    "reference_gics_sector": row["gics_sector"],
                    "reference_gics_sub_industry": row["gics_sub_industry"],
                    "reference_as_of_date": row["as_of_date"],
                    **{column: "" for column in ENTRY_COLUMNS},
                    "entry_status": "pending_authorized_source",
                    "review_notes": "",
                }
            )
    return len(rows)


def read_worksheet(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(WORKSHEET_COLUMNS) - set(reader.fieldnames or []))
        extra = sorted(set(reader.fieldnames or []) - set(WORKSHEET_COLUMNS))
        if missing or extra:
            raise ValueError(f"Worksheet columns differ: missing={missing}, extra={extra}")
        rows = [
            {column: (row.get(column) or "").strip() for column in WORKSHEET_COLUMNS}
            for row in reader
        ]
    if not rows:
        raise ValueError("Licensed-universe worksheet is empty")
    return rows


def licensed_rows_from_worksheet(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    pending: list[str] = []
    licensed_rows: list[dict[str, str]] = []
    for worksheet_row in rows:
        ticker = worksheet_row["reference_ticker"] or "<unknown>"
        licensed_row = {column: worksheet_row[f"licensed_{column}"] for column in LICENSED_COLUMNS}
        missing = [column for column, value in licensed_row.items() if not value]
        if missing:
            pending.append(f"{ticker}: {', '.join(missing)}")
        licensed_rows.append(licensed_row)
    if pending:
        preview = "; ".join(pending[:8])
        suffix = f"; plus {len(pending) - 8} more" if len(pending) > 8 else ""
        raise ValueError(f"Licensed fields remain incomplete: {preview}{suffix}")
    return licensed_rows


def reject_secondary_identifiers(rows: list[dict[str, str]]) -> None:
    invalid: list[str] = []
    for row in rows:
        ticker = row["ticker"]
        if row["security_id"].startswith("SP100-20200102:"):
            invalid.append(f"{ticker}: synthetic candidate security_id")
        if row["issuer_id"].startswith("ISSUER:"):
            invalid.append(f"{ticker}: synthetic candidate issuer_id")
        if row["source_record_id"].startswith("wikipedia:"):
            invalid.append(f"{ticker}: secondary-source record ID")
    if invalid:
        raise ValueError(
            "Candidate identifiers cannot establish licensed provenance: " + "; ".join(invalid[:8])
        )


def finalize_worksheet(candidate_path: Path, worksheet_path: Path, output_path: Path) -> int:
    candidate = read_candidate(candidate_path)
    worksheet_rows = read_worksheet(worksheet_path)
    if [row["reference_ticker"] for row in worksheet_rows] != [row["ticker"] for row in candidate]:
        raise ValueError("Worksheet reference tickers differ from the frozen candidate universe")

    licensed_rows = licensed_rows_from_worksheet(worksheet_rows)
    reject_secondary_identifiers(licensed_rows)

    # Reuse the production parser against a temporary file before touching the
    # gate input. This checks dates, sectors, required values, and identifiers.
    with tempfile.TemporaryDirectory() as temporary:
        validation_path = Path(temporary) / "licensed.csv"
        with validation_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LICENSED_COLUMNS)
            writer.writeheader()
            writer.writerows(licensed_rows)
        validated = read_licensed_universe(validation_path)

    reconciliation = reconcile_licensed_universe(candidate, validated, "2020-01-02")
    if reconciliation["status"] != "pass":
        raise ValueError(
            "Licensed universe does not reconcile; inspect ticker/GICS/as-of differences: "
            + json.dumps(reconciliation, sort_keys=True)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LICENSED_COLUMNS)
        writer.writeheader()
        writer.writerows(validated)
    return len(validated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a blank local population worksheet")
    create.add_argument(
        "--candidate-json",
        type=Path,
        default=Path("data/raw/universe_sp100_2020-01-02.json"),
    )
    create.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/licensed/universe_population_worksheet.csv"),
    )

    finalize = subparsers.add_parser(
        "finalize", help="Validate completed licensed fields and write the gate input"
    )
    finalize.add_argument(
        "--candidate-json",
        type=Path,
        default=Path("data/raw/universe_sp100_2020-01-02.json"),
    )
    finalize.add_argument(
        "--worksheet",
        type=Path,
        default=Path("data/interim/licensed/universe_population_worksheet.csv"),
    )
    finalize.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/licensed/universe_sp100_2020-01-02.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create":
        count = create_worksheet(args.candidate_json, args.output)
        print(f"Created {args.output} with {count} candidate rows; licensed fields are blank.")
    else:
        count = finalize_worksheet(args.candidate_json, args.worksheet, args.output)
        print(f"Wrote validated licensed-universe gate input to {args.output} ({count} rows).")


if __name__ == "__main__":
    main()
