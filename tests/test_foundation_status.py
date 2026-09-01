import unittest

from scripts.update_foundation_status import aggregate_foundation_status


class FoundationStatusTests(unittest.TestCase):
    def setUp(self):
        self.access = {"overall_status": "conditional_pass"}
        self.universe = {
            "status": "pass",
            "blockers": [],
            "public_universe_sha256": "universe",
            "source_manifest_sha256": "source",
        }
        self.market = {"status": "pass", "public_manifest_sha256": "market"}
        self.construction = {
            "gate_status": "pass",
            "universe_provenance_status": "pass",
            "output_scope": "confirmed_foundation_input",
            "input_hashes": {
                "universe_json_sha256": "universe",
                "universe_source_manifest_sha256": "source",
                "market_input_manifest_sha256": "market",
                "data_config_sha256": "data_config",
                "security_master_sha256": "security_master",
                "manual_corporate_actions_sha256": "actions",
                "lifecycle_events_sha256": "lifecycle",
                "observation_reviews_sha256": "reviews",
            },
            "output_hashes": {
                "portfolio_simple_return_panel_sha256": "panel",
                "active_universe_sha256": "active",
                "final_universe_sha256": "final",
            },
        }
        self.arithmetic = {
            "status": "pass",
            "inputs": {
                "simple_return_panel": {"sha256": "panel"},
                "active_universe": {"sha256": "active"},
                "final_universe": {"sha256": "final"},
            },
        }
        self.methods = {"protocol_status": "frozen_before_out_of_sample_modelling"}

    def aggregate(self, current_file_hashes=None):
        return aggregate_foundation_status(
            self.access,
            self.universe,
            self.market,
            self.construction,
            self.arithmetic,
            self.methods,
            current_file_hashes,
        )

    def test_all_independent_gates_are_required(self):
        result = self.aggregate()
        self.assertEqual(result["foundation_v2"]["status"], "pass")
        self.assertEqual(result["gates"]["modelling_readiness"]["status"], "pass")
        self.assertEqual(
            result["gates"]["data_access_feasibility_historical"]["status"],
            "conditional_pass",
        )

    def test_missing_licensed_universe_blocks_modelling(self):
        self.universe = {
            "status": "blocked",
            "blockers": ["authorized_point_in_time_universe_extract_absent"],
            "public_universe_sha256": "universe",
            "source_manifest_sha256": "source",
        }
        result = self.aggregate()
        self.assertEqual(result["foundation_v2"]["status"], "blocked")
        self.assertEqual(
            result["foundation_v2"]["blockers"],
            ["authorized_point_in_time_universe_extract_absent"],
        )
        self.assertEqual(result["gates"]["modelling_readiness"]["status"], "blocked")

    def test_failed_arithmetic_is_an_independent_blocker(self):
        self.arithmetic = {"status": "fail", "inputs": {}}
        result = self.aggregate()
        self.assertIn("portfolio_arithmetic_not_passed", result["foundation_v2"]["blockers"])

    def test_stale_arithmetic_audit_cannot_bless_rebuilt_panels(self):
        self.arithmetic["inputs"]["simple_return_panel"]["sha256"] = "stale"
        result = self.aggregate()
        self.assertIn(
            "portfolio_arithmetic_not_bound_to_current_panels",
            result["foundation_v2"]["blockers"],
        )

    def test_current_disk_hashes_are_part_of_the_gate(self):
        current = {
            "universe_json_sha256": "universe",
            "universe_source_manifest_sha256": "source",
            "market_input_manifest_sha256": "market",
            "data_config_sha256": "data_config",
            "security_master_sha256": "security_master",
            "manual_corporate_actions_sha256": "actions",
            "lifecycle_events_sha256": "lifecycle",
            "observation_reviews_sha256": "reviews",
            "portfolio_simple_return_panel_sha256": "panel",
            "active_universe_sha256": "active",
            "final_universe_sha256": "final",
        }
        self.assertEqual(self.aggregate(current)["foundation_v2"]["status"], "pass")
        current["market_input_manifest_sha256"] = "changed"
        result = self.aggregate(current)
        self.assertIn(
            "market_manifest_changed_since_integrity_audit",
            result["foundation_v2"]["blockers"],
        )
        self.assertIn(
            "construction_inputs_or_outputs_changed_since_audit",
            result["foundation_v2"]["blockers"],
        )

    def test_stale_construction_cannot_be_blessed_after_universe_passes(self):
        self.construction["universe_provenance_status"] = "blocked"
        self.construction["output_scope"] = "provisional_pending_universe_provenance"
        result = self.aggregate()
        self.assertIn(
            "data_construction_not_bound_to_approved_universe",
            result["foundation_v2"]["blockers"],
        )


if __name__ == "__main__":
    unittest.main()
