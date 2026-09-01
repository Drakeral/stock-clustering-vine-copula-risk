import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.populate_licensed_universe import (
    ENTRY_COLUMNS,
    WORKSHEET_COLUMNS,
    create_worksheet,
    finalize_worksheet,
    read_worksheet,
)


def candidate_row(ticker: str = "ABC") -> dict[str, str]:
    return {
        "security_id": f"SP100-20200102:{ticker}",
        "issuer_id": f"ISSUER:{ticker}",
        "ticker": ticker,
        "share_class": "common",
        "company_name": "ABC Corp",
        "gics_sector": "Industrials",
        "gics_sub_industry": "Machinery",
        "membership_date": "2019-11-21",
        "as_of_date": "2020-01-02",
        "source_record_id": f"wikipedia:{ticker}",
    }


class LicensedUniversePopulationTests(unittest.TestCase):
    def write_candidate(self, root: Path) -> Path:
        path = root / "candidate.json"
        path.write_text(json.dumps({"constituents": [candidate_row()]}), encoding="utf-8")
        return path

    def test_create_keeps_all_licensed_fields_blank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worksheet = root / "worksheet.csv"
            count = create_worksheet(self.write_candidate(root), worksheet)
            rows = read_worksheet(worksheet)
        self.assertEqual(count, 1)
        self.assertEqual(rows[0]["reference_ticker"], "ABC")
        self.assertTrue(all(rows[0][column] == "" for column in ENTRY_COLUMNS))
        self.assertEqual(rows[0]["entry_status"], "pending_authorized_source")

    def test_finalize_writes_only_valid_reconciled_vendor_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self.write_candidate(root)
            worksheet = root / "worksheet.csv"
            output = root / "licensed.csv"
            create_worksheet(candidate, worksheet)
            with worksheet.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            licensed = {
                "security_id": "VENDOR-SEC-1",
                "issuer_id": "VENDOR-ISSUER-1",
                "ticker": "ABC",
                "share_class": "Common Stock",
                "company_name": "ABC Corporation",
                "gics_sector": "Industrials",
                "gics_sub_industry": "Machinery",
                "membership_date": "2010-01-04",
                "as_of_date": "2020-01-02",
                "source_record_id": "vendor-row-1",
            }
            rows[0].update({f"licensed_{key}": value for key, value in licensed.items()})
            rows[0]["entry_status"] = "entered_from_authorized_source"
            with worksheet.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=WORKSHEET_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            count = finalize_worksheet(candidate, worksheet, output)
            with output.open("r", encoding="utf-8", newline="") as handle:
                final = list(csv.DictReader(handle))
        self.assertEqual(count, 1)
        self.assertEqual(final, [licensed])

    def test_finalize_rejects_candidate_identifiers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = self.write_candidate(root)
            worksheet = root / "worksheet.csv"
            create_worksheet(candidate, worksheet)
            with worksheet.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            row = candidate_row()
            rows[0].update({f"licensed_{key}": value for key, value in row.items()})
            with worksheet.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=WORKSHEET_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "cannot establish licensed provenance"):
                finalize_worksheet(candidate, worksheet, root / "licensed.csv")


if __name__ == "__main__":
    unittest.main()
