"""Read-only local IJMS dashboard. Run with the project Python environment."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import threading
import time
from urllib.parse import urlsplit, parse_qs

from spine_sim.small_array_campaign import read_config, describe

PAGE = Path(__file__).resolve().parents[1] / "monitor" / "dist" / "index.html"
RESOURCE_COMMAND = r"""
$p = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
  [pscustomobject]@{pid=$_.ProcessId; parent=$_.ParentProcessId; command=$_.CommandLine;
    cpu=([double]$_.KernelModeTime+[double]$_.UserModeTime)/10000000; rss=[double]$_.WorkingSetSize}
});
$o=Get-CimInstance Win32_OperatingSystem;
$c=Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'";
@{processes=$p; cpu_percent=$c.PercentProcessorTime; ram_total=[double]$o.TotalVisibleMemorySize*1024;
  ram_available=[double]$o.FreePhysicalMemory*1024} | ConvertTo-Json -Depth 4 -Compress
"""


class Monitor:
    def __init__(self, root: Path):
        self.root = root.resolve()
        frozen = self.root / "scan_config.json"
        self.config = read_config(frozen)["config"] if frozen.exists() else read_config()
        self.plan = describe(self.config)
        self.cache = {}
        self.previous_cpu = {}
        self.previous_time = time.monotonic()
        self.history = deque(maxlen=720)
        self.snapshot = {"loading": True}

    def resources(self):
        raw = subprocess.run(["powershell.exe", "-NoProfile", "-Command", RESOURCE_COMMAND],
                             capture_output=True, text=True, check=True, timeout=15,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        result = json.loads(raw.stdout)
        processes = result.pop("processes") or []
        selected = {p["pid"] for p in processes if any(
            marker in (p["command"] or "") for marker in
            ("run_ijms_small.py", "run_active_representative.py", "check_preload_descent.py",
             "probe_first_solve.py", "diagnose_ijms", "run_ijms_trend.py", "run_ijms_balanced_contact.py", "run_ijms_balanced_batch.py", "run_ijms_production.py"))}
        while True:
            children = {p["pid"] for p in processes if p["parent"] in selected}
            if children <= selected:
                break
            selected |= children
        now = time.monotonic()
        active = []
        parents = {p["parent"] for p in processes}
        for p in processes:
            if p["pid"] not in selected:
                continue
            previous = self.previous_cpu.get(p["pid"])
            percent = None if previous is None else max(0., (p["cpu"]-previous)/(now-self.previous_time)*100)
            active.append(dict(pid=p["pid"], ram_bytes=p["rss"], cpu_percent=percent,
                               role="worker" if ("spawn_main" in (p["command"] or "")
                                                 or (p["pid"] not in parents and "resource_tracker" not in (p["command"] or "")))
                               else "runner"))
        self.previous_cpu = {p["pid"]: p["cpu"] for p in processes if p["pid"] in selected}
        self.previous_time = now
        result.update(processes=active, logical_cpus=os.cpu_count(), disk_free=shutil.disk_usage(self.root).free)
        return result

    def results(self):
        execution, physics = Counter(), Counter()
        materials = {}
        summed_time = 0.
        for folder in (self.root / "results").glob("*"):
            if not folder.is_dir():
                continue
            database = folder / "case_summaries.sqlite3"
            if database.exists():
                wal = database.with_name(database.name + "-wal")
                stamp = (database.stat().st_mtime_ns, wal.stat().st_mtime_ns if wal.exists() else 0)
                old = self.cache.get(str(database))
                if old is None or old[0] != stamp:
                    with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=1) as connection:
                        rows = connection.execute("""SELECT run_state,
                            coalesce(json_extract(summary_json,'$.status'),'NO_PHYSICS_RESULT'),
                            count(*), sum(wall_time_s) FROM case_summary GROUP BY 1,2""").fetchall()
                    self.cache[str(database)] = (stamp, rows)
                rows = self.cache[str(database)][1]
            else:
                groups = {}
                for path in (folder / "paths").glob("*/summary.json"):
                    stamp = path.stat().st_mtime_ns
                    old = self.cache.get(str(path))
                    if old is None or old[0] != stamp:
                        row = json.loads(path.read_text(encoding="utf-8"))
                        self.cache[str(path)] = (stamp, (row["run_state"], row.get("status", "NO_PHYSICS_RESULT"),
                                                        1, row.get("wall_time_s", 0.)))
                    row = self.cache[str(path)][1]
                    key = row[:2]
                    count, elapsed = groups.get(key, (0, 0.))
                    groups[key] = (count+1, elapsed+row[3])
                rows = [(*key, *value) for key, value in groups.items()]
            material = next((m["subtype"] for m in self.config["materials"] if m["subtype"] in folder.name), "other")
            group = materials.setdefault(material, dict(recorded=0, completed=0, numerical=0, other=0))
            for run_state, status, count, elapsed in rows:
                execution[run_state] += count
                physics[status] += count
                summed_time += elapsed or 0.
                group["recorded"] += count
                category = "completed" if status == "COMPLETED" else "numerical" if status.startswith("NUMERICAL") else "other"
                group[category] += count
        return dict(recorded=sum(execution.values()), execution=dict(execution), physics=dict(physics),
                    materials=materials, summed_case_seconds=summed_time)

    def progress(self):
        latest = {}
        curves = {}
        errors = []
        for path in sorted((self.root / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime):
            size = path.stat().st_size
            with path.open("rb") as stream:
                stream.seek(max(0, size-65536))
                tail = stream.read().decode("utf-8", errors="replace")
            if "stderr" in path.name and tail.strip():
                errors.append(dict(file=path.name, tail=tail[-1600:]))
            for line in tail.splitlines():
                if not line.startswith('{"case_id":'):
                    continue
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                value["log"] = path.name
                value["last_log_age_s"] = max(0., time.time()-path.stat().st_mtime)
                latest[value["case_id"]] = value
                if value.get("status") == "STEP_ACCEPTED":
                    old_points = curves.get(value["case_id"], [])
                    if old_points and old_points[-1].get("pid") != value.get("pid"):
                        curves[value["case_id"]] = []
                    curves.setdefault(value["case_id"], []).append(value)
        curves = {key: points for key, points in curves.items()
                  if points[-1].get("pid") == latest[key].get("pid")}
        return list(latest.values())[-24:], errors[-3:], curves

    def update(self):
        if self.root.name == 'balanced_contact' or (self.root/'batch_status.json').exists():
            batch_path=self.root/'batch_status.json'
            if batch_path.exists():
                batch=json.loads(batch_path.read_text(encoding='utf-8'))
                resources=self.resources()
                self.snapshot=dict(kind='balanced',batch=batch,running=bool(resources['processes']),
                                   updated_at=datetime.now(timezone.utc).isoformat())
                return
            cases=[]
            for path in sorted((self.root/'results').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True):
                result=json.loads(path.read_text(encoding='utf-8'))
                rows=result.get('rows',[]);drag=[r for r in rows if r['phase']=='drag']
                cases.append(dict(id=path.name,configuration=result['configuration'],rigid=result['rigid'],
                                  model=result['model'],status=result['status'],solve_s=result['solve_s'],
                                  coverage_mm=rows[-1]['x_m']*1000 if rows else 0,
                                  mean_T_N=sum(r['T_N'] for r in drag)/len(drag) if drag else None))
            progress=[]
            for path in sorted((self.root/'logs').glob('*.jsonl'),key=lambda p:p.stat().st_mtime,reverse=True)[:4]:
                with path.open('rb') as stream:
                    stream.seek(max(0,path.stat().st_size-12000));lines=stream.read().splitlines()
                if lines:
                    try:progress.append(dict(json.loads(lines[-1]),file=path.name))
                    except ValueError:pass
            self.snapshot=dict(kind='balanced',cases=cases,progress=progress,resources=self.resources(),
                               root=str(self.root),updated_at=datetime.now(timezone.utc).isoformat())
            return
        trend = self.root / "trend_status.json"
        if trend.exists():
            self.snapshot = dict(json.loads(trend.read_text(encoding="utf-8")),
                                 resources=self.resources(), root=str(self.root),
                                 updated_at=datetime.now(timezone.utc).isoformat())
            return
        result = self.results()
        resources = self.resources()
        progress, errors, curves = self.progress()
        now = time.time()
        self.history.append(dict(time=now, recorded=result["recorded"], cpu=resources["cpu_percent"],
                                 ram=(resources["ram_total"]-resources["ram_available"])/2**30))
        origin = self.history[0]
        elapsed = now-origin["time"]
        delta = result["recorded"]-origin["recorded"]
        rate = delta/elapsed*3600 if elapsed >= 60 and delta >= 5 else None
        note_path = self.root / "logs" / "operator_status.json"
        note = json.loads(note_path.read_text(encoding="utf-8")) if note_path.exists() else {}
        self.snapshot = dict(updated_at=datetime.now(timezone.utc).isoformat(), root=str(self.root),
                             plan=self.plan, config=self.config, results=result, resources=resources,
                             progress=progress, errors=errors, curves=curves, history=list(self.history), note=note,
                             surfaces=len(list((self.root / "surfaces").glob("*.npy"))),
                             shards=len(list((self.root / "campaigns").glob("*.json"))),
                             cases_per_hour=rate,
                             eta_hours=(self.plan["cases"]-result["recorded"])/rate if rate else None)

    def loop(self):
        while True:
            try:
                self.update()
            except Exception as exc:
                self.snapshot = dict(self.snapshot, refresh_error=f"{type(exc).__name__}: {exc}")
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("E:/TestData/IJMS"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    monitor = Monitor(args.output_dir)
    monitor.update()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == '/api/balanced-curve':
                selected=parse_qs(url.query).get('case',[])
                known={row['id'] for row in monitor.snapshot.get('cases',[])}
                curves=[]
                for name in selected[:2]:
                    if name not in known:continue
                    result=json.loads((monitor.root/'results'/name).read_text(encoding='utf-8'))
                    curves.append(dict(name=name,rows=[{k:r[k] for k in ['x_m','T_N','P_N','Y_m','active']} for r in result['rows'] if r['phase']=='drag']))
                payload=json.dumps(curves,allow_nan=False).encode('utf-8')
                content_type='application/json; charset=utf-8'
            elif url.path == "/api/trend-summary":
                path = monitor.root / "results" / "summary.json"
                payload = path.read_bytes() if path.exists() else b'[]'
                content_type = "application/json; charset=utf-8"
            elif url.path == "/api/trend-curve":
                import numpy as np
                query = parse_qs(url.query)
                rows = json.loads((monitor.root / "results" / "summary.json").read_text(encoding="utf-8"))
                selected = [r for r in rows if r['design'] == query.get('design', [''])[0]
                            and str(r['preload_N']) == query.get('preload', [''])[0]]
                curves = []
                for row in selected:
                    with np.load(monitor.root / "results" / row['trace_file']) as trace:
                        i = row['trace_index']
                        curves.append(dict(material=row['material'], x_mm=(trace['x_m']*1000).tolist(),
                                           T_N=trace['T_N'][i].tolist(),
                                           active=trace['active_count'][i].tolist(),
                                           flagged=trace['range_flag'][i].tolist()))
                payload = json.dumps(curves, allow_nan=False).encode('utf-8')
                content_type = "application/json; charset=utf-8"
            elif self.path == "/api/status":
                payload = json.dumps(monitor.snapshot, ensure_ascii=False, allow_nan=False).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            elif self.path in ("/", "/index.html"):
                page = PAGE.with_name('trend.html') if (monitor.root / 'trend_status.json').exists() else PAGE
                if monitor.root.name=='balanced_contact' or (monitor.root/'batch_status.json').exists():page=PAGE.with_name('progress.html')
                payload = page.read_bytes()
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    threading.Thread(target=monitor.loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"IJMS monitor: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
