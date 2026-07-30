from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FINAL_SCREEN_IDS = [
    {
        "design_id": "m3_full_design_b1417f0abdd7ab61b2b1",
        "role": "优势候选A1",
        "mechanism": "刚性快速起载/中等阵列",
        "reason": "稳定起载成功率100%，中位起载距离0.2 mm；用于检验刚性快速承载。",
    },
    {
        "design_id": "m3_full_design_4f0c3f82336e077e90bc",
        "role": "优势候选A2",
        "mechanism": "刚性高累计承载/沿+x五列",
        "reason": "正向累计承载和净承载均高，起载中位距离0.5 mm。",
    },
    {
        "design_id": "m3_full_design_60bc3108fbb407b1815c",
        "role": "优势候选A3",
        "mechanism": "高刚度柔顺/大阵列/固定80°",
        "reason": "无支撑缺口、净承载为正，兼顾持续性和载荷共享。",
    },
    {
        "design_id": "m3_full_design_84bc8db4f9902f13b75c",
        "role": "优势候选A4",
        "mechanism": "低刚度柔顺/高均载",
        "reason": "300 N/m中表现突出，起载成功率高、均载好，用于检验柔顺增益。",
    },
    {
        "design_id": "m3_full_design_10af3f726401a9407f30",
        "role": "优势候选A5",
        "mechanism": "中刚度/角度梯度/小阵列",
        "reason": "800 N/m、60→80°梯度的高成功率代表，制造规模小。",
    },
    {
        "design_id": "m3_full_design_165e033132586fff1c48",
        "role": "机理对照C1",
        "mechanism": "最小刚性阵列",
        "reason": "2×2刚性仍可快速承载；用于区分阵列冗余与刚性起载效应。",
    },
    {
        "design_id": "m3_full_design_acc60e377b0cca874e29",
        "role": "机理对照C2",
        "mechanism": "刚性角度梯度",
        "reason": "与刚性固定角方案比较角度梯度对起载和恢复的影响。",
    },
    {
        "design_id": "m3_full_design_679df696f2811efa84f4",
        "role": "刚度对照C3",
        "mechanism": "匹配几何/300 N/m",
        "reason": "2×2、6 mm、固定60°、d0.8/r50刚度三联对照之一。",
    },
    {
        "design_id": "m3_full_design_5b14632013cae856bf1f",
        "role": "刚度对照C4",
        "mechanism": "匹配几何/800 N/m",
        "reason": "2×2、6 mm、固定60°、d0.8/r50刚度三联对照之一。",
    },
    {
        "design_id": "m3_full_design_373480a2081564f5a91e",
        "role": "刚度对照C5",
        "mechanism": "匹配几何/2000 N/m",
        "reason": "2×2、6 mm、固定60°、d0.8/r50刚度三联对照之一。",
    },
    {
        "design_id": "m3_full_design_ddb66bd524680d765f5c",
        "role": "方向对照C6",
        "mechanism": "2×5梯度阵列",
        "reason": "与5×2完全匹配，用于隔离阵列长边方向效应。",
    },
    {
        "design_id": "m3_full_design_7c8a201bbcbd0989e2b1",
        "role": "方向对照C7",
        "mechanism": "5×2梯度阵列",
        "reason": "与2×5完全匹配，用于隔离阵列长边方向效应。",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in ("", None) else float("nan")


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def factor_index(
    rows: list[dict[str, str]],
) -> dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, str]]:
    result = {}
    dimension_fields = (
        "spring_label",
        "angle_label",
        "array_shape",
        "spacing_mm",
        "diameter_mm",
        "tip_radius_um",
        "terrain_stratum",
        "preload_N",
    )
    for row in rows:
        levels = tuple(
            (field, row[field])
            for field in dimension_fields
            if row.get(field, "") != ""
        )
        result[(row["scope"], levels)] = row
    return result


def select_factor(
    rows: list[dict[str, str]],
    scope: str,
) -> list[dict[str, str]]:
    return [row for row in rows if row["scope"] == scope]


def design_description(row: dict[str, str]) -> str:
    angle = row["angle_label"].replace("fixed_", "固定").replace(
        "60_to_80", "60→80°"
    )
    if angle.startswith("固定"):
        angle += "°"
    return (
        f"{row['array_shape']}，{number(row, 'spacing_mm'):g} mm，"
        f"{angle}，d{number(row, 'diameter_mm'):g} mm，"
        f"r{number(row, 'tip_radius_um'):g} μm，"
        f"{row['spring_label']} N/m"
        if row["spring_label"] != "rigid"
        else (
            f"{row['array_shape']}，{number(row, 'spacing_mm'):g} mm，"
            f"{angle}，d{number(row, 'diameter_mm'):g} mm，"
            f"r{number(row, 'tip_radius_um'):g} μm，刚性"
        )
    )


def build_terminal_rows(
    designs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_id = {row["design_id"]: row for row in designs}
    output = []
    for proposal in FINAL_SCREEN_IDS:
        row = by_id[proposal["design_id"]]
        merged: dict[str, Any] = dict(proposal)
        merged.update(row)
        output.append(merged)
    return output


def write_terminal_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "role",
        "mechanism",
        "design_id",
        "array_shape",
        "spacing_mm",
        "angle_label",
        "diameter_mm",
        "tip_radius_um",
        "spring_label",
        "establish25_success_rate",
        "establish25_median_mm_successes",
        "establish25_q90_mm_successes",
        "duty25_median",
        "positive_force_ratio_mean_clipped2_median",
        "net_force_ratio_mean_median",
        "bearing_neff_fraction_median",
        "bearing_max_load_share_median",
        "case_any_gap_rate",
        "station_gap_fraction_median",
        "reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_report(
    archive_root: Path,
    design_rows: list[dict[str, str]],
    factor_rows: list[dict[str, str]],
    terminal_rows: list[dict[str, Any]],
) -> str:
    spring_rows = select_factor(factor_rows, "all_by_spring")
    spring_order = {"300": 0, "800": 1, "2000": 2, "rigid": 3}
    spring_rows.sort(key=lambda row: spring_order[row["spring_label"]])
    spring_table = markdown_table(
        [
            "轴向支撑",
            "稳定起载成功率",
            "起载距离中位数/mm",
            "承载占空比",
            "稳健正向承载/Fx÷P",
            "净承载/Fx÷P",
            "承载时Neff/N",
            "最大载荷份额",
            "有缺口case",
            "典型缺口站点",
        ],
        [
            [
                row["spring_label"],
                pct(number(row, "establish25_success_rate")),
                f(number(row, "establish25_median_mm_successes"), 2),
                pct(number(row, "duty25_median")),
                f(number(row, "positive_force_ratio_mean_clipped2_median")),
                f(number(row, "net_force_ratio_mean_median")),
                f(number(row, "bearing_neff_fraction_median")),
                f(number(row, "bearing_max_load_share_median")),
                pct(number(row, "case_any_gap_rate")),
                pct(number(row, "station_gap_fraction_median")),
            ]
            for row in spring_rows
        ],
    )

    flexible_angle = select_factor(
        factor_rows, "flexible_only_by_angle_label"
    )
    angle_order = {
        "fixed_60": 0,
        "fixed_70": 1,
        "fixed_80": 2,
        "60_to_80": 3,
    }
    flexible_angle.sort(key=lambda row: angle_order[row["angle_label"]])
    angle_table = markdown_table(
        [
            "角度",
            "稳定起载成功率",
            "起载距离/mm",
            "占空比",
            "正向承载/Fx÷P",
            "净承载/Fx÷P",
            "Neff/N",
            "最大份额",
        ],
        [
            [
                row["angle_label"],
                pct(number(row, "establish25_success_rate")),
                f(number(row, "establish25_median_mm_successes"), 2),
                pct(number(row, "duty25_median")),
                f(number(row, "positive_force_ratio_mean_clipped2_median")),
                f(number(row, "net_force_ratio_mean_median")),
                f(number(row, "bearing_neff_fraction_median")),
                f(number(row, "bearing_max_load_share_median")),
            ]
            for row in flexible_angle
        ],
    )

    rigid_angle = [
        row
        for row in select_factor(factor_rows, "spring_by_angle")
        if row["spring_label"] == "rigid"
    ]
    rigid_angle.sort(key=lambda row: angle_order[row["angle_label"]])
    rigid_angle_table = markdown_table(
        ["刚性角度", "稳定起载成功率", "起载距离/mm", "占空比", "净承载/Fx÷P"],
        [
            [
                row["angle_label"],
                pct(number(row, "establish25_success_rate")),
                f(number(row, "establish25_median_mm_successes"), 2),
                pct(number(row, "duty25_median")),
                f(number(row, "net_force_ratio_mean_median")),
            ]
            for row in rigid_angle
        ],
    )

    flexible_shape = select_factor(
        factor_rows, "flexible_only_by_array_shape"
    )
    flexible_shape.sort(key=lambda row: row["array_shape"])
    shape_table = markdown_table(
        ["阵列", "成功率", "起载/mm", "占空比", "正向承载", "净承载", "Neff/N", "最大份额"],
        [
            [
                row["array_shape"],
                pct(number(row, "establish25_success_rate")),
                f(number(row, "establish25_median_mm_successes"), 2),
                pct(number(row, "duty25_median")),
                f(number(row, "positive_force_ratio_mean_clipped2_median")),
                f(number(row, "net_force_ratio_mean_median")),
                f(number(row, "bearing_neff_fraction_median")),
                f(number(row, "bearing_max_load_share_median")),
            ]
            for row in flexible_shape
        ],
    )

    geometry_tables = []
    for scope, label, level_field in (
        ("flexible_only_by_spacing_mm", "间距/mm", "spacing_mm"),
        ("flexible_only_by_diameter_mm", "杆径/mm", "diameter_mm"),
        ("flexible_only_by_tip_radius_um", "针尖半径/μm", "tip_radius_um"),
    ):
        rows = select_factor(factor_rows, scope)
        rows.sort(key=lambda row: number(row, level_field))
        geometry_tables.append(
            (
                label,
                markdown_table(
                    [label, "成功率", "起载/mm", "占空比", "正向承载", "净承载"],
                    [
                        [
                            f(number(row, level_field), 1),
                            pct(number(row, "establish25_success_rate")),
                            f(number(row, "establish25_median_mm_successes"), 2),
                            pct(number(row, "duty25_median")),
                            f(number(row, "positive_force_ratio_mean_clipped2_median")),
                            f(number(row, "net_force_ratio_mean_median")),
                        ]
                        for row in rows
                    ],
                ),
            )
        )

    terrain_rows = select_factor(factor_rows, "all_by_terrain_stratum")
    terrain_rows.sort(key=lambda row: row["terrain_stratum"])
    terrain_table = markdown_table(
        ["地形层", "成功率", "起载/mm", "占空比", "正向承载", "净承载"],
        [
            [
                row["terrain_stratum"],
                pct(number(row, "establish25_success_rate")),
                f(number(row, "establish25_median_mm_successes"), 2),
                pct(number(row, "duty25_median")),
                f(number(row, "positive_force_ratio_mean_clipped2_median")),
                f(number(row, "net_force_ratio_mean_median")),
            ]
            for row in terrain_rows
        ],
    )

    preload_rows = select_factor(factor_rows, "all_by_preload_N")
    preload_rows.sort(key=lambda row: number(row, "preload_N"))
    preload_table = markdown_table(
        [
            "预载/N",
            "归一化稳定起载成功率",
            "起载/mm",
            "正向平均力/N",
            "净平均力/N",
            "正向功/mJ",
            "净功/mJ",
        ],
        [
            [
                f(number(row, "preload_N"), 1),
                pct(number(row, "establish25_success_rate")),
                f(number(row, "establish25_median_mm_successes"), 2),
                f(number(row, "positive_force_N_mean_clipped_2P_median")),
                f(number(row, "net_force_N_mean_median")),
                f(number(row, "positive_resisting_work_mJ_clipped_2P_median")),
                f(number(row, "net_work_mJ_median")),
            ]
            for row in preload_rows
        ],
    )

    terminal_table = markdown_table(
        [
            "角色",
            "构型",
            "成功率",
            "起载/mm",
            "占空比",
            "正向承载",
            "净承载",
            "Neff/N",
            "缺口case/典型站点",
        ],
        [
            [
                row["role"],
                design_description(row),
                pct(number(row, "establish25_success_rate")),
                f(number(row, "establish25_median_mm_successes"), 2),
                pct(number(row, "duty25_median")),
                f(number(row, "positive_force_ratio_mean_clipped2_median")),
                f(number(row, "net_force_ratio_mean_median")),
                f(number(row, "bearing_neff_fraction_median")),
                (
                    f"{pct(number(row, 'case_any_gap_rate'))}/"
                    f"{pct(number(row, 'station_gap_fraction_median'))}"
                ),
            ]
            for row in terminal_rows
        ],
    )

    matched_ids = {
        "m3_full_design_679df696f2811efa84f4",
        "m3_full_design_5b14632013cae856bf1f",
        "m3_full_design_373480a2081564f5a91e",
    }
    matched_rows = [row for row in design_rows if row["design_id"] in matched_ids]
    matched_rows.sort(key=lambda row: number(row, "spring_stiffness_N_per_m"))
    matched_table = markdown_table(
        ["刚度/N·m⁻¹", "成功率", "起载/mm", "占空比", "正向承载", "净承载"],
        [
            [
                row["spring_label"],
                pct(number(row, "establish25_success_rate")),
                f(number(row, "establish25_median_mm_successes"), 2),
                pct(number(row, "duty25_median")),
                f(number(row, "positive_force_ratio_mean_clipped2_median")),
                f(number(row, "net_force_ratio_mean_median")),
            ]
            for row in matched_rows
        ],
    )

    orientation_ids = {
        "m3_full_design_ddb66bd524680d765f5c",
        "m3_full_design_7c8a201bbcbd0989e2b1",
    }
    orientation_rows = [
        row for row in design_rows if row["design_id"] in orientation_ids
    ]
    orientation_rows.sort(key=lambda row: row["array_shape"])
    orientation_table = markdown_table(
        ["方向", "成功率", "起载/mm", "占空比", "正向承载", "净承载", "Neff/N"],
        [
            [
                row["array_shape"],
                pct(number(row, "establish25_success_rate")),
                f(number(row, "establish25_median_mm_successes"), 2),
                pct(number(row, "duty25_median")),
                f(number(row, "positive_force_ratio_mean_clipped2_median")),
                f(number(row, "net_force_ratio_mean_median")),
                f(number(row, "bearing_neff_fraction_median")),
            ]
            for row in orientation_rows
        ],
    )

    geometry_text = "\n\n".join(
        f"### {label}\n\n{table}" for label, table in geometry_tables
    )

    return f"""# M3独立阵列细筛：承载建立、维持与构型筛选报告

**版本：** v1
**数据日期：** 2026-07-30
**模型语义：** `constant-preload-reseat-v3`
**用途：** 两篇论文的共同数据底稿；一篇偏承载/再压合机理，另一篇偏阵列构型筛选。

## 摘要

本报告重新分析M3细筛的完整力—位移路径，不再把“无缺口完成率”作为刚性阵列的淘汰条件。细筛包含96个构型、150个地形条件、0.5/1/2 N三档恒定法向预载，共43,200个case；每个case记录0–10 mm、步长0.1 mm的101个空间站，并保存阵列与逐针完整路径。

核心结论是：**刚性阵列存在更高的间歇支撑比例，但其稳定起载更快、正向累计承载和净抗拖承载更高。** 69.3%的刚性case至少出现过一个未接受站点，但典型case只有2.0%的站点缺口；把“case是否出现过任一缺口”直接当成失败会严重放大问题。相反，刚性阵列在主阈值`Fx/P ≥ 0.25`下的稳定起载成功率为73.1%，成功case的中位起载距离为0.8 mm，正向承载占空比中位数为51.5%，均优于300和800 N/m总体组。2000 N/m组的起载成功率最高（81.5%），且几乎无支撑缺口，是持续承载基线。

因此后续不应沿用“先按缺口删刚性、再按Fx低分位排名”的流程，而应把构型分为快速间歇承载、连续高刚度柔顺承载、低刚度均载和方向/角度对照四类，并同时评价起载距离、承载占空比、累计正向功、净功、失载后恢复距离和载荷共享。

## 1. 数据范围与完整性

- 细筛状态：`complete`；
- 构型数：96；
- 地形条件：150；
- 预载：0.5、1.0、2.0 N；
- case数：43,200/43,200；
- 路径长度：10 mm；
- 空间步长：0.1 mm；
- 每case空间站：101；
- 全路径：已保存；
- 逐针力、法向力、切向力、接触模式、弹簧分支：已保存；
- 求解器日志：无stderr报错。

原始结果位于`01_原始数据/full_scan`。派生case表不覆盖原始NPZ或Parquet，只在`02_派生数据`中增加分析字段。

## 2. 力方向与承载定义

源码在接受空间站记录：

```text
force_x_N = -sum(wall_force_x_on_spines)
```

因此`force_x_N > 0`表示阵列对+x拖动提供阻力，`force_x_N < 0`表示地形法向分量在该站点对+x方向产生助推。以前仅使用`Fx_q10`会把负向地形区段、未建立区段和间歇恢复混在一起，无法区分“承载建立得快但偶尔丢失”和“全程没有建立有效抗拖承载”。

### 2.1 本报告的主承载阈值

主阈值定义为：

```text
Fx / preload ≥ 0.25
```

同时保留0.10和0.50两个灵敏度阈值，以及0.25 N、0.50 N两个固定绝对力阈值。

### 2.2 稳定起载距离

稳定起载距离不是第一次单点越阈，而是第一个满足以下条件的窗口起点：

- 窗口长度0.5 mm；
- 6个离散站点中至少5个达到阈值；
- 未接受站点按零承载处理，而不是从序列中删除。

这一定义允许一次局部波动，同时避免把单站几何尖峰误判为建立承载。

### 2.3 维持、恢复和累计承载

- **承载占空比：** 全路径达到主阈值的站点比例；
- **最长低承载距离：** 建立承载后连续低于阈值的最长距离；
- **恢复距离：** 能够再次达到阈值的低承载区间长度；
- **正向累计承载：** `mean(max(Fx/P,0))`；
- **稳健正向累计承载：** `mean(clip(Fx/P,0,2))`，避免少量准静态尖峰支配积分；
- **净承载：** `mean(Fx/P)`，保留负向助推区段；
- **正向功与净功：** 对10 mm路径积分，单位mJ。

## 3. 全局力—距离特征

![归一化反力距离曲线](../04_图表/图1_弹簧类型_归一化反力距离曲线.png)

从总体中位曲线看，反力并不存在统一的“从零单调爬升到平台”过程。很多case在起点预载完成后已经产生切向反力，之后随地形在正负之间波动。因此工程上的“搜索距离”应解释为**首次进入可持续抗拖窗口的距离**，而不是拟合统一的上升时间常数。

![绝对反力距离曲线](../04_图表/图1b_弹簧类型_绝对反力距离曲线.png)

绝对反力随预载提高而增加，但`Fx/P`并不保持严格线性；2 N组的归一化起载成功率下降，说明增加预载同时改变接触几何、分载和负向法向投影，不能只按比例外推。

## 4. 刚性与柔顺阵列

{spring_table}

![起载和累计承载比较](../04_图表/图2_弹簧类型_起载距离与累计承载.png)

### 4.1 刚性阵列

刚性阵列不是“不能工作”，而是表现为**快起载、高正向承载、高载荷集中、少量站点间歇失配**：

- 稳定起载成功率73.1%；
- 成功case中位起载距离0.8 mm；
- 主阈值占空比51.5%；
- 稳健正向累计承载0.337 P；
- 净承载0.318 P；
- 69.3%的case至少有一个缺口，但典型缺口只占2.0%的站点；
- 承载站点的最大载荷份额中位数为1.0，说明常由单针主承载。

这类构型适合“快速遇到凸起并立即承载”的工程目标，但需要在结构上处理单针过载和局部冲击风险。当前模型没有损伤和动力学，不能根据准静态峰值直接判断寿命。

### 4.2 2000 N/m

2000 N/m具有最高的总体稳定起载成功率（81.5%），中位起载距离1.6 mm，几乎没有支撑缺口；净承载为正，是目前最稳妥的持续承载组。其缺点是载荷共享仍不均匀，且部分地形/预载层的稳定起载概率明显下降。

### 4.3 300与800 N/m

300 N/m均载最好，但总体净承载为负；这不是“完全无承载”，因为其正向占空比和正向功仍然存在，而是负向地形投影在全路径积分中占优。800 N/m位于300与2000 N/m之间。柔顺阵列的价值主要体现在均载和连续接触，不能仅按净Fx判断。

## 5. 安装角与角度梯度

### 5.1 柔顺阵列

{angle_table}

固定80°在柔顺组中建立承载最快、成功率最高、净承载为正，同时最大载荷份额较低。60→80°梯度位于固定70°与固定80°之间，保留较高均载，但起载距离略长。固定60°总体最弱，但这一结论受当前地形坡度和坐标方向影响，不应外推为所有墙面上的普遍规律。

### 5.2 刚性阵列

{rigid_angle_table}

刚性固定60°和70°表现突出，而刚性固定80°虽然存在较高正向峰值和正向积分，但只有2.5%的case形成稳定0.5 mm承载窗口。这是典型的“峰值高、稳定承载差”机制，必须通过起载窗口和占空比与其他刚性角度分开。

## 6. 阵列形状、间距、杆径和针尖

以下表格只统计柔顺阵列，避免刚性失配机制混入几何主效应。

### 6.1 阵列形状

{shape_table}

2×2和6×6的起载成功率都较高，但含义不同：2×2的`Neff/N`高是因为分母小，6×6的净承载为正且最大载荷份额更低。不能把归一化有效针数直接当作绝对有效针数；论文中应同时报告`Neff`和`Neff/N`。

{geometry_text}

综合柔顺组：

- 5 mm间距是较稳健的折中；4 mm接近，6 mm总体稍弱但均载比例较高；
- 0.6 mm杆径明显优于0.8 mm的起载成功率和净承载；
- 100 μm针尖明显优于50 μm的稳定起载和净承载；
- 这些是边际汇总趋势，仍需通过匹配构型或配对地形比较确认因果。

## 7. 地形和预载趋势

### 7.1 地形层

{terrain_table}

P240几乎在起点即可进入稳定承载窗口，P60最难形成稳定窗口；混凝土和红砖的中位搜索距离分别约2.0和1.8 mm。由于红砖、混凝土尚未用项目实测表面标定，本报告只把这些差异解释为合成地形统计结构造成的相对趋势。

### 7.2 预载

{preload_table}

按固定比例阈值评价时，预载从0.5 N增加到2 N，稳定起载成功率下降；但绝对正向承载力仍增加。说明“更大预载”不能只用`Fx/P`评价，终筛必须同时报告：

1. 固定比例阈值的起载；
2. 固定绝对力阈值的起载；
3. 绝对正向功；
4. 归一化净功。

## 8. 匹配构型证据

边际汇总可能受构型分布影响，因此本报告保留两组严格匹配对照。

### 8.1 刚度三联对照

三者均为2×2、6 mm、固定60°、d0.8 mm、r50 μm，仅弹簧刚度不同：

{matched_table}

匹配结果显示刚度提高可明显提高稳定起载概率和净承载；这一结论比所有构型混合后的弹簧主效应更适合作为论文机理证据。

### 8.2 阵列方向对照

两者均为5 mm、60→80°、d0.6 mm、r100 μm、2000 N/m，仅2×5与5×2方向不同：

{orientation_table}

这组数据差异很小，说明在该参数包和10 mm路径下，阵列长边方向不是强主效应。这个“近似无差异”结果同样有论文价值，可防止把粗筛中的方向差异误当成普遍规律。

## 9. 终筛方案

建议终筛不是直接使用自动排名的24个构型，而是运行以下12个机理平衡构型：

{terminal_table}

完整ID和指标见`02_派生数据/终筛12构型方案.csv`。

### 9.1 终筛计算规模

建议使用：

- 构型：12个；
- 地形：全部300个现有地形条件；
- 预载：0.5、1、2 N；
- 路径：10 mm；
- 步长：0.05 mm；
- case数：12 × 300 × 3 = 10,800；
- 每case空间站：201；
- 全部保存阵列与逐针路径。

预计计算量约为本次细筛的一半；按照本机本次细筛17.4分钟估算，终筛约需9–15分钟，写盘和地形缓存可能使时间略有变化。

### 9.2 终筛判断顺序

不建立单一综合分数，依次比较：

1. `Fx/P ≥ 0.25`稳定起载成功率；
2. 成功case的起载距离中位数和90分位；
3. `Fx/P ≥ 0.25`承载占空比；
4. 正向功与净功；
5. 建立后最长低承载区和恢复距离；
6. 承载状态下`Neff`、`Neff/N`和最大载荷份额；
7. 支撑缺口长度，但不把“出现过一次缺口”设为淘汰门槛；
8. 对每个地形ID和预载做配对差值，而不是把case当成独立样本。

### 9.3 最终保留规则

从12个构型中最终保留6个：

- 3个优势构型：至少覆盖“刚性快速承载”“2000 N/m连续承载”“柔顺均载”三种机制；
- 3个对照构型：从刚度三联、2×5/5×2方向对照、刚性角度对照中选择；
- 对照构型必须能够建立承载，不以排名最低或数值失败作为选择理由。

## 10. 两篇论文的数据分工建议

### 论文一：恒定预载下爪刺阵列承载建立与再压合机制

建议主线：

- 正向抗拖反力的坐标定义；
- 稳定起载窗口定义；
- 刚性阵列的快速承载与间歇支撑；
- 柔顺阵列的连续接触和载荷共享；
- 刚度三联匹配对照；
- 失载后恢复距离与正向/净功。

推荐主图：图1、图1b、图2，以及终筛后的刚度三联逐构型曲线。

### 论文二：阵列几何、角度和针刺参数的构型筛选

建议主线：

- 固定60/70/80°与+x方向60→80°；
- 2×5/5×2方向匹配对照；
- 2×2至6×6规模与载荷集中；
- 4/5/6 mm间距；
- 0.6/0.8 mm杆径和50/100 μm针尖；
- 3种优势构型与3种可制造机理对照。

推荐主图：图3、终筛12构型的配对地形箱线图和最终6构型的力—距离包络。

## 11. 统计与论文作图建议

- 地形ID作为配对单位，预载作为重复条件；
- 主结果报告中位数、10/90分位和配对差值；
- 对构型差值使用地形层内配对bootstrap置信区间；
- 不把101个空间站当作101个独立统计样本；
- 对红砖、混凝土、砂纸分别报告，再给总体分层汇总；
- 峰值图同时给原始值和稳健截顶值，避免少量几何尖峰支配坐标轴；
- 所有论文图保留正负Fx，不用绝对值掩盖方向反转。

## 12. 模型边界

1. 当前是准静态降阶模型，不包含惯性、冲击、磨损、断裂和地形演化；
2. 刚性阵列的未接受站点记录为数值平衡失败，不能直接等同于现实脱落；
3. 准静态峰值不是冲击峰值；
4. 红砖、混凝土地形仍是未完成实测标定的合成表面；
5. 细筛结果用于相对构型选择，不用于声明真实墙面的绝对承载；
6. 当前路径10 mm足以比较起载和短程维持，但不能替代寿命或长距离磨损实验。

## 13. 归档索引

- `01_原始数据/full_scan`：粗筛和细筛原始输出；
- `02_派生数据/case_bearing_metrics.parquet`：43,200个case的承载建立指标；
- `02_派生数据/design_bearing_metrics.csv`：96个构型汇总；
- `02_派生数据/factor_bearing_metrics.csv`：参数主效应与交互汇总；
- `02_派生数据/terrain_preload_bearing_metrics.csv`：地形×预载×弹簧汇总；
- `02_派生数据/force_distance_curves_by_spring_preload.csv`：力—距离曲线；
- `02_派生数据/终筛12构型方案.csv`：建议终筛集合；
- `04_图表`：本报告图件；
- `05_复现脚本`：派生指标、报告和工作簿生成脚本；
- `99_校验/SHA256SUMS.csv`：归档文件校验值。
"""


def build_readme(archive_root: Path) -> str:
    return f"""# M3独立阵列细筛数据归档

本目录归档2026-07-30完成的M3独立阵列粗筛、细筛原始输出和承载建立再分析。

## 快速入口

- 详细报告：`03_报告/M3_独立阵列细筛_承载建立与构型筛选报告_v1.md`
- 分析工作簿：`03_报告/M3_细筛论文分析数据_v1.xlsx`
- 96构型汇总：`02_派生数据/design_bearing_metrics.csv`
- 43,200 case派生表：`02_派生数据/case_bearing_metrics.parquet`
- 终筛方案：`02_派生数据/终筛12构型方案.csv`
- 原始结果：`01_原始数据/full_scan`
- 哈希清单：`99_校验/SHA256SUMS.csv`

## 地形数据说明

没有复制96 GB的M1地形矩阵库。归档只保存地形目录和生成报告元数据，
原地形母库仍位于：

`D:\\Code\\Spine_Sim\\results\\m1_material_formal_300\\terrain_library`

细筛原始数据中的`terrain_table.parquet`保存了每个地形条件的ID、seed、
材料层、源路径和SHA-256，可用于追溯。

## 数据不可变约定

`01_原始数据`为只读快照。所有新指标只写入`02_派生数据`，不改写NPZ、
Parquet、manifest或原始选择文件。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    arguments = parser.parse_args()
    archive_root = arguments.archive_root.resolve()
    derived = archive_root / "02_派生数据"
    report_dir = archive_root / "03_报告"
    report_dir.mkdir(parents=True, exist_ok=True)

    designs = read_csv(derived / "design_bearing_metrics.csv")
    factors = read_csv(derived / "factor_bearing_metrics.csv")
    terminal_rows = build_terminal_rows(designs)
    write_terminal_csv(derived / "终筛12构型方案.csv", terminal_rows)
    report = build_report(archive_root, designs, factors, terminal_rows)
    (report_dir / "M3_独立阵列细筛_承载建立与构型筛选报告_v1.md").write_text(
        report,
        encoding="utf-8",
    )
    (archive_root / "00_说明与索引" / "README.md").write_text(
        build_readme(archive_root),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
