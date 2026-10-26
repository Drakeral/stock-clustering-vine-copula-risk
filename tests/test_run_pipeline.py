from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_pipeline  # noqa: E402


class PipelineRegistryTests(unittest.TestCase):
    def test_reproduce_profile_has_one_unique_ordered_path(self) -> None:
        names = [stage.name for stage in run_pipeline.REPRODUCE_STAGES]
        self.assertEqual(len(names), len(set(names)))
        self.assertLess(names.index("foundation-gate"), names.index("core-groupings"))
        self.assertLess(names.index("core-evaluation"), names.index("ml-groupings"))
        self.assertEqual(names[-2:], ["run-manifest", "lineage"])

    def test_validation_profile_covers_all_release_checks(self) -> None:
        names = [stage.name for stage in run_pipeline.VALIDATE_STAGES]
        self.assertEqual(len(names), len(set(names)))
        for required in (
            "lint",
            "complexity",
            "security",
            "format",
            "compile",
            "tests",
            "foundation-gate",
            "report-artifacts",
            "run-manifest",
            "lineage",
        ):
            self.assertIn(required, names)

    def test_select_stages_is_inclusive_and_rejects_reverse_bounds(self) -> None:
        chosen = run_pipeline.select_stages(run_pipeline.VALIDATE_STAGES, "compile", "tests")
        self.assertEqual([stage.name for stage in chosen], ["compile", "tests"])
        with self.assertRaisesRegex(ValueError, "occurs after"):
            run_pipeline.select_stages(run_pipeline.VALIDATE_STAGES, "tests", "compile")
        with self.assertRaisesRegex(ValueError, "Unknown start stage"):
            run_pipeline.select_stages(run_pipeline.VALIDATE_STAGES, "missing", None)

    def test_commands_use_current_python_and_project_local_scripts(self) -> None:
        for stage in run_pipeline.REPRODUCE_STAGES:
            command = stage.command("/safe/python")
            self.assertEqual(command[0], "/safe/python")
            if command[1] != "-m":
                script = run_pipeline.PROJECT_ROOT / command[1]
                self.assertEqual(script.parent, SCRIPTS)
                self.assertTrue(script.is_file(), script)

    @mock.patch("run_pipeline.subprocess.run")
    def test_runner_is_fail_fast_and_never_uses_a_shell(self, run: mock.Mock) -> None:
        run.side_effect = [None, subprocess.CalledProcessError(2, ["python", "bad.py"])]
        stages = run_pipeline.VALIDATE_STAGES[:3]
        with (
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            run_pipeline.run_stages(stages)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertNotIn("shell", call.kwargs)
            self.assertTrue(call.kwargs["check"])
            self.assertEqual(call.kwargs["cwd"], run_pipeline.PROJECT_ROOT)

    @mock.patch("run_pipeline.subprocess.run")
    def test_dry_run_executes_nothing(self, run: mock.Mock) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            run_pipeline.run_stages(run_pipeline.VALIDATE_STAGES, dry_run=True)
        run.assert_not_called()

    def test_cli_defaults_to_validation_and_lists_without_execution(self) -> None:
        args = run_pipeline.parse_args([])
        self.assertEqual(args.profile, "validate")
        self.assertFalse(args.dry_run)
        with (
            mock.patch("run_pipeline.run_stages") as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = run_pipeline.main(["--profile", "reproduce", "--list-stages"])
        self.assertEqual(result, 0)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
