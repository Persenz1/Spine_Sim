from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spine_sim.core.config import BaseCaseSpec, CampaignSpec
from spine_sim.io.results import (
    CompactResultStore,
    ResultStore,
    atomic_write_json,
    open_result_store,
    read_trace_table,
)
from spine_sim.runtime.backend import BackendConfig, discover_backend
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
            self.assertTrue(store.is_incomplete("case_interrupted"))

    def test_read_only_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ResultStore(temporary)
            store.write_case(
                case_id="case_x",
                config={"x": 1},
                summary={"run_state": "complete"},
                arrays={"x": np.array([1.0])},
                trace_rows=(
                    {
                        "path_position_m": 0.0,
                        "accepted": True,
                        "nested": {"parameter_sources": {}},
                    },
                ),
                complete=True,
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
                summary={"run_state": "complete"},
                events=({"label": "diagnostic"},),
                complete=True,
            )
            event_file = store.case_dir("case_hash") / "events.jsonl"
            self.assertTrue(event_file.is_file())
            self.assertTrue(store.is_complete("case_hash"))
            event_file.write_text("tampered\n", encoding="utf-8")
            self.assertFalse(store.is_complete("case_hash"))
            store.write_case(
                case_id="case_hash",
                config={"x": 1},
                summary={"run_state": "complete"},
                events=(),
                complete=True,
            )
            self.assertFalse(event_file.exists())
            self.assertTrue(store.is_complete("case_hash"))
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

    def test_cpu_only_backend_and_forced_gpu_recording(self) -> None:
        cpu = discover_backend(BackendConfig(preference="cpu"))
        self.assertTrue(cpu.cpu_available)
        self.assertEqual(cpu.selected, "cpu")
        previous = os.environ.get("SPINE_SIM_FORCE_CUDA")
        os.environ["SPINE_SIM_FORCE_CUDA"] = "1"
        try:
            cuda = discover_backend(BackendConfig(preference="auto"))
            self.assertTrue(cuda.cuda_available)
            self.assertEqual(cuda.selected, "cuda")
            self.assertEqual(cuda.cuda_provider, "environment_override")
        finally:
            if previous is None:
                os.environ.pop("SPINE_SIM_FORCE_CUDA", None)
            else:
                os.environ["SPINE_SIM_FORCE_CUDA"] = previous

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
                "spine_sim.examples.fake_module:run_summary_case",
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
