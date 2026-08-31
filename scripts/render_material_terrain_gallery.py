"""用 M1 绘图接口渲染代表性的 5 mm 材料地形画廊。

每种表面同时输出三维斜视图、二维高度图、原始 NPZ 和可追踪的地形库记录；
可选 HTML 只是图片排版，不参与物理计算。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from spine_sim.terrain import (
    generate_terrain,
    register_terrain,
    render_terrain_views,
    save_terrain,
)


SURFACES = (
    ("sandpaper", "P40", 4029, "砂纸 P40"),
    ("sandpaper", "P100", 10029, "砂纸 P100"),
    ("red_brick", "fired_brick_standard", 22029, "烧结红砖"),
    ("concrete", "rough_wall", 33029, "粗糙混凝土"),
)
SIZE_M = 5e-3
RESOLUTION_M = 10e-6


def _write_visualization(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """复制画廊图片到 HTML 旁，并生成可独立嵌入的响应式片段。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    for record in records:
        # 使用相对图片名，使整个 HTML 目录移动后仍能显示。
        asset_stem = str(record["stem"])
        three_d_name = f"{asset_stem}-3d-oblique.png"
        two_d_name = f"{asset_stem}-2d-heightmap.png"
        shutil.copyfile(record["three_d_path"], path.parent / three_d_name)
        shutil.copyfile(record["two_d_path"], path.parent / two_d_name)
        figures.append(
            f"""
  <section class="terrain-pair" aria-label="{record['label']}地形图">
    <h3>{record['label']}</h3>
    <p class="text-small text-muted">5 mm × 5 mm · 10 µm 网格 · seed {record['seed']} · RMS {record['rms_height_um']:.1f} µm</p>
    <div class="terrain-view-grid">
      <figure>
        <img src="{three_d_name}" alt="{record['label']}三维斜视地形，带1毫米标尺">
        <figcaption class="text-small">3D 斜视图 · z 轴视觉放大 {record['z_exaggeration']:.2f}×</figcaption>
      </figure>
      <figure>
        <img src="{two_d_name}" alt="{record['label']}二维高度图，带1毫米标尺">
        <figcaption class="text-small">2D 高度图 · 色标单位 µm</figcaption>
      </figure>
    </div>
  </section>"""
        )
    # CSS 限定在组件 id 下，避免嵌入现有页面后污染全局样式。
    fragment = f"""<div id="material-terrain-gallery">
  <style>
    #material-terrain-gallery {{
      display: grid;
      gap: 28px;
      color: var(--foreground);
    }}
    #material-terrain-gallery .terrain-pair {{
      display: grid;
      gap: 8px;
    }}
    #material-terrain-gallery h3,
    #material-terrain-gallery p,
    #material-terrain-gallery figure {{
      margin: 0;
    }}
    #material-terrain-gallery .terrain-view-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      align-items: start;
    }}
    #material-terrain-gallery img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    #material-terrain-gallery figcaption {{
      margin-top: 5px;
      color: var(--muted-foreground);
      text-align: center;
    }}
    @media (max-width: 620px) {{
      #material-terrain-gallery .terrain-view-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
{''.join(figures)}
</div>
"""
    path.write_text(fragment, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """生成四种代表表面，渲染双视图并写出统一 manifest。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualization-html", type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    library_root = output / "terrain_library"
    records: list[dict[str, Any]] = []
    for material, subtype, seed, label in SURFACES:
        # 1. 用冻结种子生成 5 mm 方形表面，保证画廊可以逐次复现。
        terrain = generate_terrain(
            material=material,
            subtype=subtype,
            size_x_m=SIZE_M,
            size_y_m=SIZE_M,
            resolution_m=RESOLUTION_M,
            seed=seed,
            mode="synthetic",
        )
        stem = f"{material}-{subtype}".replace("_", "-")
        # 2. NPZ 便于独立交换；地形库注册则供规范渲染 API 按身份读取。
        artifact = save_terrain(output / f"{stem}.npz", terrain)
        recipe, region, _ = register_terrain(
            library_root,
            terrain,
            purpose="debug",
            overwrite=True,
        )
        view_output = output / stem
        # 3. 所有表面共享视窗、球半径和采样上限，保证视觉比较公平。
        views = render_terrain_views(
            library_root,
            recipe.terrain_recipe_id,
            region.region_id,
            view_output,
            overview_size_m=SIZE_M,
            sphere_radius_m=100e-6,
            overview_maximum_axis_points=601,
            surface_maximum_axis_points=181,
            dpi=190,
            prefix=stem,
        )
        metadata = json.loads(
            Path(views["metadata_path"]).read_text(encoding="utf-8")
        )
        values_um = np.asarray(terrain.height, dtype=np.float64) * 1e6
        # 4. 统计量直接由未降采样高度图计算，渲染抽样不会影响数值。
        records.append(
            {
                "material": material,
                "subtype": subtype,
                "label": label,
                "stem": stem,
                "seed": seed,
                "profile_status": terrain.metadata["profile_status"],
                "shape_yx": list(terrain.height.shape),
                "resolution_m": terrain.dx,
                "size_x_m": terrain.size_x_m,
                "size_y_m": terrain.size_y_m,
                "z_exaggeration": metadata["rendering"][
                    "oblique_vertical_exaggeration"
                ],
                "minimum_height_um": float(np.min(values_um)),
                "maximum_height_um": float(np.max(values_um)),
                "rms_height_um": float(np.std(values_um)),
                "terrain_artifact": str(artifact),
                "three_d_path": views["files"]["oblique_3d"],
                "two_d_path": views["files"]["heightmap_2d"],
            }
        )
    # manifest 是机器可读索引，HTML 仅在用户显式指定时生成。
    manifest = {
        "schema_version": "material-terrain-gallery-v1",
        "renderer": "spine_sim.terrain.render_terrain_views",
        "size_x_m": SIZE_M,
        "size_y_m": SIZE_M,
        "resolution_m": RESOLUTION_M,
        "surfaces": records,
    }
    manifest_path = output / "gallery_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.visualization_html is not None:
        _write_visualization(args.visualization_html.resolve(), records)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "visualization_html": (
                    None
                    if args.visualization_html is None
                    else str(args.visualization_html.resolve())
                ),
                "surface_count": len(records),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
