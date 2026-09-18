from __future__ import annotations

from dataclasses import replace
import json
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from spine_sim.cli import build_parser, main as cli_main
from spine_sim.core.config import BaseCaseSpec, CampaignSpec
from spine_sim.io.files import atomic_write_json
from spine_sim.io.results import (
    CompactResultStore,
    ResultStore,
    open_result_store,
    read_trace_table,
)
from spine_sim.core.errors import ConfigurationError
from spine_sim.core.states import Event, EventType, PhysicalState
from spine_sim.runtime.backend import (
    BackendConfig,
    discover_backend,
    validate_environment,
)
from spine_sim.runtime.runner import CaseOutput, CampaignRunner, _execute


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


def make_persistence_failure_campaign(*, workers: int) -> CampaignSpec:
    cases = (
        BaseCaseSpec("m0", "fake-1", {"seed": 1, "samples": 5}),
        BaseCaseSpec(
            "m0",
            "fake-1",
            {"seed": 2, "samples": 5, "invalid_persisted_event": True},
        ),
        BaseCaseSpec("m0", "fake-1", {"seed": 3, "samples": 5}),
    )
    return CampaignSpec(
        "persistence-failure",
        "m0-test",
        "spine_sim.examples.fake_module:run_case",
        cases,
        workers,
    )


def make_hard_exit_campaign() -> CampaignSpec:
    return CampaignSpec(
        "hard-exit",
        "m0-test",
        "spine_sim.examples.fake_module:run_case",
        tuple(
            BaseCaseSpec(
                "m0",
                "fake-1",
                (
                    {"seed": seed, "hard_exit": True}
                    if seed == 2
                    else {"seed": seed, "samples": 5}
                ),
            )
            for seed in range(10)
        ),
        workers=2,
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

    def test_trace_falls_back_to_jsonl_without_optional_pyarrow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            sys.modules, {"pyarrow": None, "pyarrow.parquet": None}
        ):
            store = ResultStore(temporary)
            store.write_case(
                case_id="case_jsonl",
                config={"x": 1},
                summary=complete_summary(),
                trace_rows=({"station": 1, "nested": {"ok": True}},),
            )
            summary = store.load_case_summary("case_jsonl")
            self.assertEqual(summary["trace_format"], "jsonl_fallback")
            self.assertEqual(summary["trace_file"], "trace.jsonl")
            self.assertEqual(
                read_trace_table(store.case_dir("case_jsonl")),
                [{"nested": {"ok": True}, "station": 1}],
            )
            self.assertTrue(store.is_complete("case_jsonl"))

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

    def test_native_solver_event_round_trips_through_result_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ResultStore(temporary)
            event = Event(
                event_type=EventType.CONTACT,
                sequence=0,
                from_state=PhysicalState.SEARCH,
                to_state=PhysicalState.CONTACT,
                case_id="case_event",
                load_parameter=0.25,
            )
            store.write_case(
                case_id="case_event",
                config={"x": 1},
                summary=complete_summary(),
                events=(event,),
            )
            event_path = store.case_dir("case_event") / "events.jsonl"
            persisted = json.loads(event_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted, event.as_dict())
            self.assertTrue(store.is_complete("case_event"))

    def test_corrupt_complete_case_is_reported_as_integrity_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ResultStore(temporary)
            store.write_case(
                case_id="case_corrupt",
                config={"x": 1},
                summary=complete_summary(),
                arrays={"x": np.array([1.0])},
                events=({"label": "must-not-be-indexed-after-corruption"},),
            )
            (store.case_dir("case_corrupt") / "path.npz").write_bytes(
                b"corrupt"
            )

            records = store.list_records()
            self.assertEqual(
                [record.run_state for record in records], ["execution_error"]
            )
            self.assertEqual(records[0].error_type, "ResultIntegrityError")
            store.rebuild_event_index()
            self.assertEqual((store.root / "events.jsonl").read_bytes(), b"")


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

    def test_campaign_index_falls_back_without_optional_pyarrow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            sys.modules, {"pyarrow": None, "pyarrow.parquet": None}
        ):
            runner = CampaignRunner(
                make_campaign(),
                temporary,
                discover_backend(BackendConfig(preference="cpu")),
            )
            runner.initialize({"test": True})
            records = runner.run(workers=1)

            self.assertEqual(
                [record.run_state for record in records],
                ["complete", "complete"],
            )
            self.assertTrue((Path(temporary) / "cases.jsonl").is_file())
            manifest = json.loads(
                (Path(temporary) / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["index_format"], "jsonl_fallback")

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

    def test_retry_failed_reruns_a_corrupt_complete_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner, records = self.run_campaign(Path(temporary), 1)
            target_id = records[0].case_id
            path = runner.store.case_dir(target_id) / "path.npz"
            path.write_bytes(b"corrupt")
            self.assertFalse(runner.store.is_complete(target_id))
            self.assertIn(
                "execution_error",
                {record.run_state for record in runner.store.list_records()},
            )

            retried = runner.run(failed_only=True)
            self.assertEqual(
                [record.run_state for record in retried],
                ["complete", "complete"],
            )
            self.assertTrue(runner.store.is_complete(target_id))

            summary = runner.store.case_dir(target_id) / "summary.json"
            summary.write_text("{", encoding="utf-8")
            repaired_again = runner.run(failed_only=True)
            self.assertEqual(
                [record.run_state for record in repaired_again],
                ["complete", "complete"],
            )
            self.assertTrue(runner.store.is_complete(target_id))

    def test_summarize_requires_the_exact_expected_case_id_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = make_campaign()
            runner = CampaignRunner(
                campaign,
                temporary,
                discover_backend(BackendConfig(preference="cpu")),
            )
            runner.initialize({"test": True})
            for case_id in ("case_extra_1", "case_extra_2"):
                runner.store.write_case(
                    case_id=case_id,
                    config={"extra": True},
                    summary=complete_summary(),
                    events=({"label": "unexpected"},),
                )

            self.assertEqual(len(runner.store.list_records()), len(campaign.cases))
            indexed = runner.run(failed_only=True)
            self.assertEqual(
                {record.run_state for record in indexed}, {"execution_error"}
            )
            self.assertEqual((runner.store.root / "events.jsonl").read_bytes(), b"")
            with patch("builtins.print") as printed:
                self.assertEqual(cli_main(["summarize", temporary]), 1)
            summary = json.loads(printed.call_args.args[0])
            self.assertFalse(summary["case_ids_match"])
            self.assertEqual(summary["missing_case_count"], 2)
            self.assertEqual(summary["unexpected_case_count"], 2)
            self.assertEqual(summary["status_counts"], {"execution_error": 2})

    def test_single_process_execute_does_not_swallow_process_control(self) -> None:
        case = make_campaign().cases[0]
        backend = discover_backend(BackendConfig(preference="cpu")).as_dict()

        for exception in (KeyboardInterrupt(), SystemExit(7)):
            with self.subTest(exception_type=type(exception).__name__):
                def interrupting_callable(_parameters, _context):
                    raise exception

                with patch(
                    "spine_sim.runtime.runner._load_callable",
                    return_value=interrupting_callable,
                ):
                    with self.assertRaises(type(exception)) as caught:
                        _execute(
                            "ignored:test_callable",
                            case,
                            backend,
                            False,
                        )
                if isinstance(exception, SystemExit):
                    self.assertEqual(caught.exception.code, 7)

    def test_execute_binds_the_selected_cupy_device(self) -> None:
        case = make_campaign().cases[0]
        device = Mock()
        device_factory = Mock(return_value=device)
        fake_cupy = SimpleNamespace(
            cuda=SimpleNamespace(Device=device_factory)
        )
        backend = {
            **discover_backend(BackendConfig(preference="cpu")).as_dict(),
            "selected": "cuda",
            "cuda_provider": "cupy",
            "device_index": 2,
        }
        with patch.dict(sys.modules, {"cupy": fake_cupy}), patch(
            "spine_sim.runtime.runner._load_callable",
            return_value=lambda _parameters, _context: CaseOutput(),
        ):
            result = _execute("ignored:callable", case, backend, False)

        self.assertTrue(result["ok"])
        device_factory.assert_called_once_with(2)
        device.use.assert_called_once_with()

    def test_persistence_failure_is_isolated_for_single_and_spawn_workers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for workers in (1, 2):
                campaign = make_persistence_failure_campaign(workers=workers)
                runner = CampaignRunner(
                    campaign,
                    Path(temporary) / str(workers),
                    discover_backend(BackendConfig(preference="cpu")),
                )
                runner.initialize({"test": True})
                records = runner.run()
                self.assertEqual(
                    [record.run_state for record in records].count("complete"),
                    2,
                )
                failures = [
                    record
                    for record in records
                    if record.run_state == "execution_error"
                ]
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0].error_type, "TypeError")

    def test_worker_hard_exit_is_indexed_instead_of_aborting_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = make_hard_exit_campaign()
            runner = CampaignRunner(
                campaign,
                temporary,
                discover_backend(BackendConfig(preference="cpu")),
            )
            runner.initialize({"test": True})
            records = runner.run()

            self.assertEqual(
                {record.case_id for record in records},
                {case.case_id for case in campaign.cases},
            )
            self.assertIn(
                "execution_error", {record.run_state for record in records}
            )
            manifest = json.loads(
                (Path(temporary) / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                sum(manifest["status_counts"].values()), len(campaign.cases)
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

    def test_backend_rejects_invalid_and_out_of_range_device_indices(self) -> None:
        for invalid in (-1, True, 1.5):
            with self.subTest(device_index=invalid):
                with self.assertRaisesRegex(
                    ConfigurationError, "non-negative integer"
                ):
                    BackendConfig(device_index=invalid)

        with patch(
            "spine_sim.runtime.backend._detect_cuda",
            return_value=(True, "cupy", ["cupy_device_count=1"]),
        ):
            with self.assertRaisesRegex(ConfigurationError, "out of range"):
                discover_backend(
                    BackendConfig(preference="cuda", device_index=1)
                )

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

    def test_environment_probes_output_writability_and_rejects_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writable = validate_environment(
                BackendConfig(preference="cpu"),
                writable_path=root / "new-results",
            )
            check = next(
                item
                for item in writable["checks"]
                if item["name"] == "results_path_writable"
            )
            self.assertTrue(check["passed"])

            nested = validate_environment(
                BackendConfig(preference="cpu"),
                writable_path=root / "missing-parent" / "nested" / "results",
            )
            check = next(
                item
                for item in nested["checks"]
                if item["name"] == "results_path_writable"
            )
            self.assertTrue(check["passed"])

            target_file = root / "not-a-directory"
            target_file.write_text("occupied", encoding="utf-8")
            blocked = validate_environment(
                BackendConfig(preference="cpu"), writable_path=target_file
            )
            check = next(
                item
                for item in blocked["checks"]
                if item["name"] == "results_path_writable"
            )
            self.assertFalse(check["passed"])
            self.assertFalse(blocked["passed"])

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

    def test_prepare_rejects_backend_and_lineage_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = make_campaign()
            cpu = discover_backend(BackendConfig(preference="cpu"))
            runner = CampaignRunner(campaign, root, cpu)
            runner.initialize({"test": True})

            cpu_with_unused_cuda_probe = replace(
                cpu,
                cuda_available=False,
                cuda_provider="cupy",
                detection_notes=("cupy_device_count=0",),
            )
            resumed_cpu = CampaignRunner(
                campaign, root, cpu_with_unused_cuda_probe
            )
            resumed_cpu.prepare({"test": True})
            stored_backend = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )["backend"]
            self.assertEqual(resumed_cpu._backend_record, stored_backend)

            cuda = replace(
                cpu,
                cuda_available=True,
                cuda_provider="cupy",
                selected="cuda",
                device_index=0,
            )
            with self.assertRaisesRegex(ConfigurationError, "backend"):
                CampaignRunner(campaign, root, cuda).prepare({"test": True})

            changed_cases = tuple(
                replace(case, tags=("changed-tag",))
                for case in campaign.cases
            )
            changed = replace(campaign, cases=changed_cases)
            self.assertEqual(changed.campaign_id, campaign.campaign_id)
            with self.assertRaisesRegex(ConfigurationError, "lineage"):
                CampaignRunner(changed, root, cpu).prepare({"test": True})

    def test_prepare_refuses_missing_lineage_after_cases_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = make_campaign()
            backend = discover_backend(BackendConfig(preference="cpu"))
            runner = CampaignRunner(campaign, root, backend)
            runner.initialize({"test": True})
            self.assertTrue(runner.run()[0].run_state == "complete")
            (root / "lineage.json").unlink()

            changed = replace(
                campaign,
                cases=tuple(
                    replace(case, tags=("new-provenance",))
                    for case in campaign.cases
                ),
            )
            self.assertEqual(changed.campaign_id, campaign.campaign_id)
            with self.assertRaisesRegex(
                ConfigurationError, "unsafe resume"
            ):
                CampaignRunner(changed, root, backend).prepare({"test": True})

    def test_prepare_refuses_missing_manifest_after_cases_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = make_campaign()
            backend = discover_backend(BackendConfig(preference="cpu"))
            runner = CampaignRunner(campaign, root, backend)
            runner.initialize({"test": True})
            runner.run()
            (root / "manifest.json").unlink()

            changed = replace(
                campaign,
                cases=tuple(
                    replace(case, tags=("new-provenance",))
                    for case in campaign.cases
                ),
            )
            with self.assertRaisesRegex(
                ConfigurationError, "unsafe initialization"
            ):
                CampaignRunner(changed, root, backend).prepare({"test": True})

    def test_prepare_repairs_interrupted_compact_initialization(self) -> None:
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
                "repair-compact",
                "m0-test",
                "spine_sim.examples.fake_module:run_case",
                cases,
                workers=1,
                mode="formal",
            )
            backend = discover_backend(BackendConfig(preference="cpu"))
            runner = CampaignRunner(campaign, temporary, backend)
            runner.initialize({"test": True})
            self.assertIsInstance(runner.store, CompactResultStore)
            runner.store.database_path.unlink()
            self.assertFalse(runner.store.is_initialized())

            recovered = CampaignRunner(campaign, temporary, backend)
            recovered.prepare({"test": True})
            self.assertTrue(recovered.store.is_initialized())
            self.assertEqual(
                [record.run_state for record in recovered.run()],
                ["complete", "complete"],
            )

    def test_runner_rejects_stale_semantic_versions_and_cuda_multiworker(
        self,
    ) -> None:
        cpu = discover_backend(BackendConfig(preference="cpu"))
        stale_case = replace(
            make_campaign().cases[0],
            solver_semantics_version="single-array-event-v1",
        )
        stale = CampaignSpec(
            "stale",
            "m0-test",
            "spine_sim.examples.fake_module:run_case",
            (stale_case,),
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ConfigurationError, "semantic versions"):
                CampaignRunner(stale, temporary, cpu)

            cuda = replace(
                cpu,
                cuda_available=True,
                cuda_provider="cupy",
                selected="cuda",
                device_index=0,
            )
            campaign = make_campaign(workers=2)
            runner = CampaignRunner(campaign, temporary, cuda)
            runner.initialize({"test": True})
            with self.assertRaisesRegex(ConfigurationError, "workers=1"):
                runner.run()

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

            connection = sqlite3.connect(runner.store.database_path)
            try:
                with connection:
                    connection.execute(
                        "UPDATE case_summary SET summary_json = summary_json || ' ' "
                        "WHERE case_id = ?",
                        (records[0].case_id,),
                    )
            finally:
                connection.close()
            self.assertIn(
                "execution_error",
                {record.run_state for record in reopened.list_records()},
            )
            repaired = runner.run(failed_only=True)
            self.assertEqual(
                [record.run_state for record in repaired],
                ["complete", "complete"],
            )


if __name__ == "__main__":
    unittest.main()
