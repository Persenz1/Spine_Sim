"""Build a compact inline M2 global-reaction-versus-distance visualization."""

from __future__ import annotations

import argparse
import base64
import json
import struct
from pathlib import Path

import numpy as np


TEMPLATE = r"""
<div id="m2-global-reaction-current">
  <style>
    #m2-global-reaction-current { color: var(--foreground); width: 100%; }
    #m2-global-reaction-current .plot-wrap { position: relative; width: 100%; }
    #m2-global-reaction-current .reaction-chart { display: block; width: 100%; height: auto; overflow: visible; }
    #m2-global-reaction-current .grid { stroke: var(--border); stroke-width: 1; }
    #m2-global-reaction-current .axis { stroke: var(--muted-foreground); stroke-width: 1; }
    #m2-global-reaction-current .curve { fill: none; stroke: var(--viz-series-1); stroke-width: 1.6; vector-effect: non-scaling-stroke; }
    #m2-global-reaction-current .reference { fill: none; stroke-width: 1.2; vector-effect: non-scaling-stroke; }
    #m2-global-reaction-current .median { stroke: var(--viz-series-2); stroke-dasharray: 6 4; }
    #m2-global-reaction-current .p10 { stroke: var(--viz-series-3); stroke-dasharray: 2 4; }
    #m2-global-reaction-current .nominal { stroke: var(--muted-foreground); stroke-dasharray: 8 4 2 4; }
    #m2-global-reaction-current .cursor { stroke: var(--foreground); stroke-width: 1; opacity: 0; pointer-events: none; }
    #m2-global-reaction-current .cursor-dot { fill: var(--viz-series-1); stroke: var(--background); stroke-width: 2; opacity: 0; pointer-events: none; }
    #m2-global-reaction-current .hit { fill: transparent; cursor: crosshair; }
    #m2-global-reaction-current .tick, #m2-global-reaction-current .axis-label, #m2-global-reaction-current .annotation { fill: var(--muted-foreground); }
    #m2-global-reaction-current .series-label { fill: var(--foreground); font-weight: 500; }
    #m2-global-reaction-current .detail { min-height: 1.5em; margin-top: 0.25rem; text-align: center; color: var(--muted-foreground); }
    #m2-global-reaction-current .tooltip { position: absolute; visibility: hidden; pointer-events: none; transform: translate(-50%, calc(-100% - 10px)); }
  </style>
  <div class="plot-wrap">
    <svg class="reaction-chart" viewBox="0 0 736 390" role="img" aria-labelledby="m2-reaction-title m2-reaction-desc">
      <title id="m2-reaction-title">全局拖曳反力随拖动距离变化</title>
      <desc id="m2-reaction-desc">seed __SEED__ 的低摩擦 M2 case；横轴为拖动距离，纵轴为全局坐标系拖曳方向反力绝对值。</desc>
      <g class="grid-layer"></g>
      <g class="reference-layer"></g>
      <path class="curve"></path>
      <g class="axis-layer"></g>
      <line class="cursor"></line>
      <circle class="cursor-dot" r="4"></circle>
      <rect class="hit"></rect>
    </svg>
    <div class="tooltip" role="status"></div>
  </div>
  <div class="detail text-small" aria-live="polite">移动指针查看曲线数值</div>
  <script>
    (() => {
      const root = document.getElementById("m2-global-reaction-current");
      const svg = root.querySelector(".reaction-chart");
      const ns = "http://www.w3.org/2000/svg";
      const bytes = Uint8Array.from(atob("__DATA_BASE64__"), c => c.charCodeAt(0));
      const view = new DataView(bytes.buffer);
      const count = __COUNT__;
      const xs = new Float32Array(count);
      const ys = new Float32Array(count);
      for (let i = 0; i < count; i += 1) {
        xs[i] = view.getFloat32(i * 8, true);
        ys[i] = view.getFloat32(i * 8 + 4, true);
      }

      const width = 736, height = 390;
      const margin = { left: 66, right: 28, top: 24, bottom: 54 };
      const plot = {
        left: margin.left,
        right: width - margin.right,
        top: margin.top,
        bottom: height - margin.bottom
      };
      const xMax = __X_MAX__;
      const yMax = __Y_AXIS_MAX__;
      const sx = x => plot.left + (x / xMax) * (plot.right - plot.left);
      const sy = y => plot.bottom - (y / yMax) * (plot.bottom - plot.top);
      const make = (name, attrs, parent) => {
        const el = document.createElementNS(ns, name);
        Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
        parent.appendChild(el);
        return el;
      };
      const grid = root.querySelector(".grid-layer");
      const axes = root.querySelector(".axis-layer");
      const refs = root.querySelector(".reference-layer");

      for (let x = 0; x <= xMax + 1e-6; x += 20) {
        make("line", { class: "grid", x1: sx(x), x2: sx(x), y1: plot.top, y2: plot.bottom }, grid);
        const label = make("text", { class: "tick text-small", x: sx(x), y: plot.bottom + 22, "text-anchor": "middle" }, axes);
        label.textContent = x.toFixed(0);
      }
      const yStep = yMax <= 1 ? 0.2 : yMax <= 2 ? 0.25 : 0.5;
      for (let y = 0; y <= yMax + 1e-9; y += yStep) {
        make("line", { class: "grid", x1: plot.left, x2: plot.right, y1: sy(y), y2: sy(y) }, grid);
        const label = make("text", { class: "tick text-small", x: plot.left - 10, y: sy(y) + 4, "text-anchor": "end" }, axes);
        label.textContent = y.toFixed(2);
      }
      make("line", { class: "axis", x1: plot.left, x2: plot.right, y1: plot.bottom, y2: plot.bottom }, axes);
      make("line", { class: "axis", x1: plot.left, x2: plot.left, y1: plot.top, y2: plot.bottom }, axes);
      const xLabel = make("text", { class: "axis-label", x: (plot.left + plot.right) / 2, y: height - 10, "text-anchor": "middle" }, axes);
      xLabel.textContent = "拖动距离 (mm)";
      const yLabel = make("text", { class: "axis-label", x: 16, y: (plot.top + plot.bottom) / 2, transform: `rotate(-90 16 ${(plot.top + plot.bottom) / 2})`, "text-anchor": "middle" }, axes);
      yLabel.textContent = "全局拖曳反力 |Fx| (N)";

      [
        { value: __MEDIAN__, cls: "median", label: "中位数 __MEDIAN_LABEL__ N" },
        { value: __P10__, cls: "p10", label: "P10 __P10_LABEL__ N" },
        { value: __NOMINAL__, cls: "nominal", label: "平面 μkW __NOMINAL_LABEL__ N" }
      ].forEach((ref, index) => {
        const y = sy(ref.value);
        make("line", { class: `reference ${ref.cls}`, x1: plot.left, x2: plot.right, y1: y, y2: y }, refs);
        const label = make("text", { class: "annotation text-small", x: plot.right - 5, y: y - 5 - index * 0, "text-anchor": "end" }, refs);
        label.textContent = ref.label;
      });

      let d = "";
      for (let i = 0; i < count; i += 1) {
        d += `${i === 0 ? "M" : "L"}${sx(xs[i]).toFixed(2)},${sy(ys[i]).toFixed(2)}`;
      }
      root.querySelector(".curve").setAttribute("d", d);
      const seriesLabel = make("text", { class: "series-label text-small", x: plot.left + 8, y: plot.top + 16 }, axes);
      seriesLabel.textContent = "seed __SEED__ · R__RADIUS__ μm · d__DIAMETER__ mm · __AXIAL__ · 80° · μs/μk=0.30/0.20";

      const hit = root.querySelector(".hit");
      hit.setAttribute("x", plot.left);
      hit.setAttribute("y", plot.top);
      hit.setAttribute("width", plot.right - plot.left);
      hit.setAttribute("height", plot.bottom - plot.top);
      const cursor = root.querySelector(".cursor");
      const dot = root.querySelector(".cursor-dot");
      const tooltip = root.querySelector(".tooltip");
      const detail = root.querySelector(".detail");

      const nearest = target => {
        let lo = 0, hi = count - 1;
        while (lo < hi) {
          const mid = Math.floor((lo + hi) / 2);
          if (xs[mid] < target) lo = mid + 1;
          else hi = mid;
        }
        if (lo > 0 && Math.abs(xs[lo - 1] - target) < Math.abs(xs[lo] - target)) return lo - 1;
        return lo;
      };
      const show = event => {
        const rect = svg.getBoundingClientRect();
        const px = (event.clientX - rect.left) * width / rect.width;
        const target = Math.max(0, Math.min(xMax, (px - plot.left) * xMax / (plot.right - plot.left)));
        const i = nearest(target);
        const cx = sx(xs[i]), cy = sy(ys[i]);
        cursor.setAttribute("x1", cx); cursor.setAttribute("x2", cx);
        cursor.setAttribute("y1", plot.top); cursor.setAttribute("y2", plot.bottom);
        cursor.style.opacity = "0.45";
        dot.setAttribute("cx", cx); dot.setAttribute("cy", cy);
        dot.style.opacity = "1";
        const leftPx = cx / width * rect.width;
        const topPx = cy / height * rect.height;
        tooltip.style.left = `${Math.max(70, Math.min(rect.width - 70, leftPx))}px`;
        tooltip.style.top = `${Math.max(42, topPx)}px`;
        tooltip.style.visibility = "visible";
        tooltip.textContent = `${xs[i].toFixed(2)} mm · ${ys[i].toFixed(3)} N`;
        detail.textContent = `拖动 ${xs[i].toFixed(2)} mm：全局拖曳反力 ${ys[i].toFixed(3)} N`;
      };
      hit.addEventListener("pointermove", show);
      hit.addEventListener("pointerleave", () => {
        cursor.style.opacity = "0";
        dot.style.opacity = "0";
        tooltip.style.visibility = "hidden";
        detail.textContent = "移动指针查看曲线数值";
      });
    })();
  </script>
</div>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((case_dir / "config.json").read_text(encoding="utf-8"))
    with np.load(case_dir / "path.npz", allow_pickle=False) as arrays:
        distance_mm = np.asarray(arrays["path_position_m"], dtype=float) * 1e3
        reaction_n = np.abs(
            np.asarray(
                arrays["spine_on_plate_wrench_about_holder"][:, 0],
                dtype=float,
            )
        )

    packed = b"".join(
        struct.pack("<ff", float(distance), float(force))
        for distance, force in zip(distance_mm, reaction_n, strict=True)
    )
    spine = config["parameters"]["spine"]
    policy = config["parameters"]["screening_policy"]
    nominal = (
        float(spine["kinetic_friction"])
        * float(config["parameters"]["experiment"]["constant_preload_n"])
    )
    y_max = max(float(np.nanmax(reaction_n)), summary["global_pull_force_median_n"])
    y_axis_max = max(0.2, np.ceil(y_max * 5.0) / 5.0)
    axial = (
        "刚性轴向"
        if spine["spring_stiffness_n_m"] is None
        else f"k={float(spine['spring_stiffness_n_m']):g} N/m"
    )
    replacements = {
        "__DATA_BASE64__": base64.b64encode(packed).decode("ascii"),
        "__COUNT__": str(len(distance_mm)),
        "__X_MAX__": f"{float(np.nanmax(distance_mm)):.8g}",
        "__Y_AXIS_MAX__": f"{float(y_axis_max):.8g}",
        "__SEED__": str(policy["seed"]),
        "__RADIUS__": f"{float(spine['tip_radius_m']) * 1e6:g}",
        "__DIAMETER__": f"{float(spine['diameter_m']) * 1e3:g}",
        "__AXIAL__": axial,
        "__MEDIAN__": f"{summary['global_pull_force_median_n']:.8g}",
        "__MEDIAN_LABEL__": f"{summary['global_pull_force_median_n']:.3f}",
        "__P10__": f"{summary['global_pull_force_p10_n']:.8g}",
        "__P10_LABEL__": f"{summary['global_pull_force_p10_n']:.3f}",
        "__NOMINAL__": f"{nominal:.8g}",
        "__NOMINAL_LABEL__": f"{nominal:.3f}",
    }
    fragment = TEMPLATE
    for old, new in replacements.items():
        fragment = fragment.replace(old, new)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment.strip() + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "case_id": summary["case_id"],
                "point_count": len(distance_mm),
                "saved_path_peak_n": float(np.nanmax(reaction_n)),
                "all_step_peak_n": summary["global_pull_force_peak_n"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
