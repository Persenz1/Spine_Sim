"""Small read-only Windows progress window for the M3 full scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk


STAGES = (
    ("coarse", "粗筛", 45),
    ("fine", "细筛", 150),
    ("final", "终筛", 300),
)


class ScanMonitor:
    def __init__(self, root: tk.Tk, output_root: Path) -> None:
        self.root = root
        self.output_root = output_root
        self.root.title("M3 全量扫描进度")
        self.root.geometry("720x430")
        self.root.minsize(660, 390)

        heading = ttk.Label(
            root,
            text="M3 独立阵列全量扫描",
            font=("Microsoft YaHei UI", 16, "bold"),
        )
        heading.pack(anchor="w", padx=20, pady=(18, 4))
        ttk.Label(
            root,
            text=str(output_root),
            foreground="#555555",
        ).pack(anchor="w", padx=20, pady=(0, 14))

        self.stage_widgets: dict[str, tuple[ttk.Progressbar, ttk.Label]] = {}
        for stage, label, _ in STAGES:
            frame = ttk.LabelFrame(root, text=label, padding=12)
            frame.pack(fill="x", padx=20, pady=6)
            progress = ttk.Progressbar(
                frame,
                orient="horizontal",
                mode="determinate",
                maximum=100.0,
            )
            progress.pack(fill="x")
            status = ttk.Label(frame, text="等待启动")
            status.pack(anchor="w", pady=(6, 0))
            self.stage_widgets[stage] = (progress, status)

        self.overall = ttk.Label(
            root,
            text="正在等待后台仿真进程……",
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        self.overall.pack(anchor="w", padx=20, pady=(12, 4))
        self.log_tail = tk.Text(
            root,
            height=5,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
        )
        self.log_tail.pack(fill="both", expand=True, padx=20, pady=(4, 16))
        self.refresh()

    def _manifest(self, stage: str) -> dict[str, object]:
        path = self.output_root / stage / "manifest.json"
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _completed_conditions(self, stage: str) -> int:
        directory = self.output_root / stage / "complete"
        if not directory.is_dir():
            return 0
        return sum(1 for _ in directory.glob("*.json"))

    def _read_log_tail(self) -> str:
        paths = (
            ("输出", self.output_root / "run.log"),
            ("错误", self.output_root / "run.err.log"),
        )
        available = [(label, path) for label, path in paths if path.is_file()]
        if not available:
            return "运行日志尚未创建。"
        tails: list[str] = []
        for label, path in available:
            try:
                lines = path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                continue
            if lines:
                tails.append(f"[{label}] " + "\n".join(lines[-4:]))
        return "\n".join(tails) or "后台任务正在运行。"

    def refresh(self) -> None:
        total_expected = 0
        total_complete = 0
        all_complete = True
        any_started = False
        for stage, _, expected in STAGES:
            manifest = self._manifest(stage)
            completed = min(self._completed_conditions(stage), expected)
            total_expected += expected
            total_complete += completed
            progress, status = self.stage_widgets[stage]
            progress["value"] = 100.0 * completed / expected
            stage_status = str(manifest.get("status", "waiting"))
            if stage_status != "complete":
                all_complete = False
            if manifest or completed:
                any_started = True
            expected_cases = manifest.get("expected_case_count")
            case_text = (
                f"，计划 {int(expected_cases):,} cases"
                if isinstance(expected_cases, (int, float))
                else ""
            )
            elapsed = manifest.get("elapsed_s")
            elapsed_text = (
                f"，耗时 {float(elapsed) / 60.0:.1f} 分钟"
                if isinstance(elapsed, (int, float))
                else ""
            )
            status.configure(
                text=(
                    f"{completed}/{expected} 个地形分片"
                    f"（{100.0 * completed / expected:.1f}%）"
                    f"{case_text}，状态：{stage_status}{elapsed_text}"
                )
            )

        if all_complete:
            overall_text = "全部仿真完成，可以关闭此窗口。"
        elif any_started:
            overall_text = (
                f"总进度：{total_complete}/{total_expected} 个地形分片"
                f"（{100.0 * total_complete / total_expected:.1f}%）"
            )
        else:
            overall_text = "正在等待后台仿真进程……"
        self.overall.configure(text=overall_text)

        self.log_tail.configure(state="normal")
        self.log_tail.delete("1.0", "end")
        self.log_tail.insert("1.0", self._read_log_tail())
        self.log_tail.configure(state="disabled")
        self.root.after(2000, self.refresh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/m3_fast/full_scan"),
    )
    args = parser.parse_args()
    root = tk.Tk()
    ScanMonitor(root, args.output.resolve())
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
