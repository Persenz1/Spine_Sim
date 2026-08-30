from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spine_sim.cli import build_parser
from spine_sim.core.config import BaseCaseSpec, CampaignSpec
from spine_sim.io.files import atomic_write_json
from spine_sim.io.results import (
    CompactResultStore,
    ResultStore,
    open_result_store,
    read_trace_table,
)
from spine_sim.core.errors import ConfigurationError
from spine_sim.runtime.backend import (
    BackendConfig,
    discover_backend,
    validate_environment,
)
from spine_sim.runtime.runner import CampaignRunner


def make_campaign(*, include_failure: bool = False, workers: int = 1) -> CampaignSpec:
    cases = [
        BaseCaseSpec("m0", "fake-1", {"seed": 1, "samples": 5}),
        BaseCaseSpec("m0", "fake-1", {"seed": 2, "samples": 5}),
    ]
    if include_failure:
        cases.append(BaseCaseSpec("m0", "fake-1", {"seed": 3, "fail": True}))
    return CampaignSpec(
        "test", "m0-test", "spine_sim.examples.fake_module:run_case", tuple(cases), workers
    )


def complete_summary() -> dict[str, object]:
    return {
        "run_state": "complete",
        "wall_time_s": 0.0,
        "peak_ram_bytes": 0,
        "peak_python_bytes": 0,
    }


class ResultTests(unittest.TestCase):
    def test_atomic_write_and_interrupted_case_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "nested" / "value.json"
            atomic_write_json(target, {"ok": True})
            self.assertEqual(json.loads(target.read_text())["ok"], True)
            self.assertFalse(any(path.suffix == ".tmp" for path in target.parent.iterdir()))
            store = ResultStore(root / "campaign")
            store.case_dir("case_interrupted").mkdir(parents=True)
            (store.case_dir("case_interrupted") / "summary.json").write_text("{", encoding="utf-8")
            self.assertFalse(store.is_complete("case_interrupted"))

    def test_case_records_require_runtime_fields_and_complete_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ResultStore(temporary)
            with self.assertRaises(KeyError):
                store.write_case(
                    case_id="missing_metrics",
                    config={},
                    summary={"run_state": "complete"},
                )
            with self.assertRaises(KeyError):
                store.write_case(
                    case_id="partial_error",
                    config={},
                    summary={
                        **complete_summary(),
                        "run_state": "execution_error",
                        "error": {"category": "execution"},
                    },
                )

    def test_open_compact_store_requires_declared_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_write_json(
                root / "manifest.json",
                {"result_storage": "sqlite_transactional_summary_v1"},
            )
            with self.assertRaisesRegex(ValueError, "SQLite database"):
                open_result_store(root)

    def test_open_store_rejects_unknown_declared_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_write_json(
                root / "manifest.json",
                {"result_storage": "legacy_directory_guess_v0"},
            )
            with self.assertRaisesRegex(ValueError, "unsupported result storage"):
                open_result_store(root)

    def test_read_only_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ResultStore(temporary)
            store.write_case(
                case_id="case_x",
                config={"x": 1},
                summary=complete_summary(),
                arrays={"x": np.array([1.0])},
                trace_rows=(
                    {
                        "path_position_m": 0.0,
                        "accepted": True,
                        "nested": {"parameter_sources": {}},
                    },
                ),
            )
            before = (store.case_dir("case_x") / "summary.json").stat().st_mtime_ns
            self.assertEqual(store.load_case_summary("case_x")["case_id"], "case_x")
            after = (store.case_dir("case_x") / "summary.json").stat().st_mtime_ns
            self.assertEqual(before, after)
            summary = store.load_case_summary("case_x")
            self.assertIn(summary["trace_format"], {"parquet", "jsonl_fallback"})
            self.assertTrue(
                (store.case_dir("case_x") / summary["trace_file"]).is_file()
            )
            self.assertEqual(
                read_trace_table(store.case_dir("case_x"))[0]["nested"],
                {"parameter_sources": {}},
            )

    def test_complete_hash_and_empty_event_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ResultStore(temporary)
            store.write_case(
                case_id="case_hash",
                config={"x": 1},
                summary=complete_summary(),
                events=({"label": "diagnostic"},),
            )
            event_file = store.case_dir("case_hash") / "events.jsonl"
            self.assertTrue(event_file.is_file())
            self.assertTrue(store.is_complete("case_hash"))
            summary_file = store.case_dir("case_hash") / "summary.json"
            atomic_write_json(summary_file, [])
            self.assertFalse(store.is_complete("case_hash"))
            store.write_case(
                case_id="case_hash",
                config={"x": 1},
                summary=complete_summary(),
                events=({"label": "diagnostic"},),
            )
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            summary["numerical_state"] = "tampered"
            atomic_write_json(summary_file, summary)
            self.assertFalse(store.is_complete("case_hash"))
            store.write_case(
                case_id="case_hash",
                config={"x": 1},
                summary=complete_summary(),
                events=({"label": "diagnostic"},),
            )
            atomic_write_json(store.case_dir("case_hash") / "config.json", {"x": 2})
            self.assertFalse(store.is_complete("case_hash"))
            store.write_case(
                case_id="case_hash",
                config={"x": 1},
                summary=complete_summary(),
                events=({"label": "diagnostic"},),
            )
            event_file.write_text("tampered\n", encoding="utf-8")
            self.assertFalse(store.is_complete("case_hash"))
            store.write_case(
                case_id="case_hash",
                config={"x": 1},
                summary=complete_summary(),
                events=(),
                validation={"status": "passed"},
            )
            self.assertFalse(event_file.exists())
            self.assertTrue(store.is_complete("case_hash"))
            atomic_write_json(
                store.case_dir("case_hash") / "validation.json",
                {"status": "tampered"},
            )
            self.assertFalse(store.is_complete("case_hash"))
            store.write_case(
                case_id="case_hash",
                config={"x": 1},
                summary=complete_summary(),
                events=(),
                validation={"status": "passed"},
            )
            (store.case_dir("case_hash") / "validation.json").unlink()
            self.assertFalse(store.is_complete("case_hash"))
            store.write_case(
                case_id="case_hash",
                config={"x": 1},
                summary=complete_summary(),
                events=(),
                validation={"status": "passed"},
            )
            marker = store.case_dir("case_hash") / "COMPLETE"
            marker.write_text("tampered\n", encoding="ascii")
            self.assertFalse(store.is_complete("case_hash"))


class RunnerTests(unittest.TestCase):
    def run_campaign(self, directory: Path, workers: int, include_failure: bool = False):
        campaign = make_campaign(include_failure=include_failure, workers=workers)
        runner = CampaignRunner(
            campaign,
            directory,
            discover_backend(BackendConfig(preference="cpu")),
        )
        runner.initialize({"test": True})
        return runner, runner.run()

    def test_single_and_spawn_multi_worker_content_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            one_runner, one_records = self.run_campaign(root / "one", 1)
            two_runner, two_records = self.run_campaign(root / "two", 2)
            one_hashes = {row.case_id: row.result_hash for row in one_records}
            two_hashes = {row.case_id: row.result_hash for row in two_records}
            self.assertEqual(one_hashes, two_hashes)
            self.assertTrue((one_runner.store.root / "events.jsonl").is_file())
            self.assertIn(
                json.loads((one_runner.store.root / "manifest.json").read_text())["index_format"],
                {"parquet", "jsonl_fallback"},
            )

    def test_failure_is_isolated_and_resume_skips_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner, records = self.run_campaign(Path(temporary), 2, include_failure=True)
            states = [row.run_state for row in records]
            self.assertEqual(states.count("complete"), 2)
            self.assertEqual(states.count("execution_error"), 1)
            complete_mtimes = {
                row.case_id: (runner.store.case_dir(row.case_id) / "summary.json").stat().st_mtime_ns
                for row in records
                if row.run_state == "complete"
            }
            runner.run(resume=True)
            self.assertEqual(
                complete_mtimes,
                {
                    case_id: (runner.store.case_dir(case_id) / "summary.json").stat().st_mtime_ns
                    for case_id in complete_mtimes
                },
            )

    def test_cpu_fallback_and_cupy_gpu_recording(self) -> None:
        cpu = discover_backend(BackendConfig(preference="cpu"))
        self.assertTrue(cpu.cpu_available)
        self.assertEqual(cpu.selected, "cpu")

        with patch(
            "spine_sim.runtime.backend._detect_cuda",
            return_value=(True, "cupy", ["cupy_device_count=1"]),
        ):
            cuda = discover_backend(BackendConfig(preference="auto"))
            self.assertTrue(cuda.cuda_available)
            self.assertEqual(cuda.selected, "cuda")
            self.assertEqual(cuda.cuda_provider, "cupy")

        with patch(
            "spine_sim.runtime.backend.importlib.util.find_spec",
            return_value=None,
        ) as find_spec:
            fallback = discover_backend(BackendConfig(preference="auto"))
        find_spec.assert_called_once_with("cupy")
        self.assertFalse(fallback.cuda_available)
        self.assertEqual(fallback.selected, "cpu")

        with patch(
            "spine_sim.runtime.backend._detect_cuda",
            return_value=(False, None, ["cupy_probe_failed:RuntimeError"]),
        ):
            with self.assertRaises(ConfigurationError):
                discover_backend(BackendConfig(preference="cuda"))

    def test_environment_report_keeps_capability_without_empty_cpu_check(
        self,
    ) -> None:
        with patch(
            "spine_sim.runtime.backend._detect_cuda",
            return_value=(False, None, []),
        ):
            report = validate_environment(BackendConfig(preference="cpu"))
        self.assertTrue(report["backend"]["cpu_available"])
        self.assertNotIn(
            "cpu_available",
            {check["name"] for check in report["checks"]},
        )

    def test_run_case_does_not_accept_ignored_workers_option(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["run-case", "campaign.json", "--workers", "2"]
            )

    def test_worker_override_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = CampaignRunner(
                make_campaign(),
                temporary,
                discover_backend(BackendConfig(preference="cpu")),
            )
            runner.initialize({"test": True})
            with self.assertRaisesRegex(ConfigurationError, "at least one"):
                runner.run(workers=0)

    def test_formal_summary_campaign_uses_compact_transactional_store(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cases = tuple(
                BaseCaseSpec(
                    "m0",
                    "fake-1",
                    {
                        "seed": seed,
                        "samples": 5,
                        "output": {"level": "summary"},
                    },
                )
                for seed in (1, 2)
            )
            campaign = CampaignSpec(
                "formal-summary-test",
                "m0-test",
                "spine_sim.examples.fake_module:run_case",
                cases,
                workers=1,
                mode="formal",
            )
            runner = CampaignRunner(
                campaign,
                Path(temporary),
                discover_backend(BackendConfig(preference="cpu")),
            )
            self.assertIsInstance(runner.store, CompactResultStore)
            runner.initialize({"test": True})
            records = runner.run()
            self.assertEqual(
                [record.run_state for record in records],
                ["complete", "complete"],
            )
            self.assertFalse((Path(temporary) / "paths").exists())
            self.assertIs(runner.store.is_complete(records[0].case_id), True)
            reopened = open_result_store(temporary)
            self.assertIsInstance(reopened, CompactResultStore)
            self.assertEqual(len(reopened.list_records()), 2)
            before = {
                row.case_id: row.result_hash for row in records
            }
            after = {
                row.case_id: row.result_hash
                for row in runner.run(resume=True)
            }
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
