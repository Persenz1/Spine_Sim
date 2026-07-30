from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PRIMARY_IDS = (
    "m3_full_design_b1417f0abdd7ab61b2b1",
    "m3_full_design_4f0c3f82336e077e90bc",
    "m3_full_design_60bc3108fbb407b1815c",
    "m3_full_design_10af3f726401a9407f30",
)

RECOMMENDATION = {
    "m3_full_design_b1417f0abdd7ab61b2b1": (
        "主工程构型",
        "快速稳定起载",
        "严格0.5 mm窗口成功率最高、起载距离最短；保留刚性单针承载风险评估。",
    ),
    "m3_full_design_4f0c3f82336e077e90bc": (
        "主工程构型",
        "高净承载",
        "正向累计承载和净承载最高；与A1形成速度—强度双目标。",
    ),
    "m3_full_design_60bc3108fbb407b1815c": (
        "主工程构型",
        "无缺口连续承载",
        "36针高刚度柔顺阵列、无支撑缺口、净承载为正。",
    ),
    "m3_full_design_10af3f726401a9407f30": (
        "主工程构型",
        "紧凑柔顺分载",
        "2×2小阵列、约3针有效分载、总体净承载略为正。",
    ),
    "m3_full_design_84bc8db4f9902f13b75c": (
        "机理保留",
        "低刚度高均载",
        "起载和占空比优于A5且分载最好，但总体净承载为负，作为柔顺边界。",
    ),
    "m3_full_design_165e033132586fff1c48": (
        "机理保留",
        "最小刚性阵列",
        "证明2×2刚性仍能承载，用于阵列冗余与单针集中讨论。",
    ),
    "m3_full_design_acc60e377b0cca874e29": (
        "机理保留",
        "刚性角度梯度",
        "稳定成功率仅次于A1，但与A1间距不同，不能作为严格角度因果对照。",
    ),
    "m3_full_design_679df696f2811efa84f4": (
        "严格对照",
        "刚度三联300 N/m",
        "与K800/K2000共同保留，不能拆开。",
    ),
    "m3_full_design_5b14632013cae856bf1f": (
        "严格对照",
        "刚度三联800 N/m",
        "与K300/K2000共同保留，不能拆开。",
    ),
    "m3_full_design_373480a2081564f5a91e": (
        "严格对照",
        "刚度三联2000 N/m",
        "与K300/K800共同保留，不能拆开。",
    ),
    "m3_full_design_ddb66bd524680d765f5c": (
        "严格对照",
        "方向2×5",
        "与D5×2共同保留，验证长边方向弱效应。",
    ),
    "m3_full_design_7c8a201bbcbd0989e2b1": (
        "严格对照",
        "方向5×2",
        "与D2×5共同保留，验证长边方向弱效应。",
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def num(value: Any, digits: int = 3) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def pair_lookup(
    paired_rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["pair_name"], row["metric"]): row for row in paired_rows
    }


def ci_text(row: dict[str, str], digits: int = 3) -> str:
    return (
        f"{num(row['mean_difference_B_minus_A'], digits)} "
        f"[{num(row['ci95_low'], digits)}, "
        f"{num(row['ci95_high'], digits)}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build M3 terminal Markdown report.")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()

    derived = args.archive_root / "02_派生数据" / "终筛"
    report_dir = args.archive_root / "03_报告"
    report_dir.mkdir(parents=True, exist_ok=True)
    designs = read_csv(derived / "terminal_design_metrics.csv")
    strata = read_csv(derived / "terminal_design_terrain_preload_metrics.csv")
    preloads = read_csv(derived / "terminal_design_preload_metrics.csv")
    terrain_preloads = read_csv(derived / "terminal_terrain_preload_metrics.csv")
    paired = read_csv(derived / "terminal_paired_comparisons.csv")
    design_by_id = {row["design_id"]: row for row in designs}
    pair = pair_lookup(paired)
    source_manifest = json.loads(
        (args.source_root / "final" / "manifest.json").read_text(encoding="utf-8")
    )
    analysis_manifest = json.loads(
        (derived / "terminal_analysis_manifest.json").read_text(encoding="utf-8")
    )

    recommendation_rows: list[dict[str, Any]] = []
    for design in designs:
        level, mechanism, reason = RECOMMENDATION[design["design_id"]]
        recommendation_rows.append(
            {
                "recommendation_level": level,
                "recommended_mechanism": mechanism,
                "role": design["role"],
                "design_id": design["design_id"],
                "array_shape": design["array_shape"],
                "spacing_mm": design["spacing_mm"],
                "angle_label": design["angle_label"],
                "diameter_mm": design["diameter_mm"],
                "tip_radius_um": design["tip_radius_um"],
                "spring_label": design["spring_label"],
                "establish25_success_rate": design[
                    "establish25_success_rate"
                ],
                "establish25_median_mm_successes": design[
                    "establish25_median_mm_successes"
                ],
                "duty25_median": design["duty25_median"],
                "positive_force_ratio_mean_clipped2_median": design[
                    "positive_force_ratio_mean_clipped2_median"
                ],
                "net_force_ratio_mean_median": design[
                    "net_force_ratio_mean_median"
                ],
                "bearing_neff_fraction_median": design[
                    "bearing_neff_fraction_median"
                ],
                "case_any_gap_rate": design["case_any_gap_rate"],
                "station_gap_fraction_median": design[
                    "station_gap_fraction_median"
                ],
                "recommendation_reason": reason,
            }
        )
    write_csv(derived / "terminal_recommendations.csv", recommendation_rows)

    design_table = md_table(
        [
            "构型",
            "主要参数",
            "0.5 mm稳定成功率",
            "起载中位/mm",
            "起载P90/mm",
            "占空比",
            "正向承载",
            "净承载",
            "Neff/N",
        ],
        [
            [
                row["role"],
                (
                    f"{row['array_shape']}，{num(row['spacing_mm'], 0)} mm，"
                    f"{row['angle_label']}，{row['spring_label']} N/m"
                ).replace("rigid N/m", "刚性"),
                pct(row["establish25_success_rate"]),
                num(row["establish25_median_mm_successes"], 2),
                num(row["establish25_q90_mm_successes"], 2),
                pct(row["duty25_median"]),
                num(row["positive_force_ratio_mean_clipped2_median"]),
                num(row["net_force_ratio_mean_median"]),
                num(row["bearing_neff_fraction_median"]),
            ]
            for row in designs
        ],
    )

    sensitivity_table = md_table(
        ["构型", "0.25 mm窗口成功率", "0.5 mm窗口成功率", "首次越阈距离中位/mm"],
        [
            [
                row["role"],
                pct(row["establish25_0p25mm_window_success_rate"]),
                pct(row["establish25_success_rate"]),
                num(row["onset25_median_mm"], 2),
            ]
            for row in designs
        ],
    )

    gap_table = md_table(
        [
            "构型",
            "有缺口case",
            "典型缺口站点",
            "再压合站点",
            "邻域换落点",
            "建立后最长低载/mm",
            "恢复距离/mm",
        ],
        [
            [
                row["role"],
                pct(row["case_any_gap_rate"]),
                pct(row["station_gap_fraction_median"]),
                pct(row["recontacted_station_fraction_median"]),
                pct(row["alternate_landing_station_fraction_median"]),
                num(row["maximum_below25_median_mm_successes"], 2),
                num(row["recovery_q90_median_mm_successes"], 2),
            ]
            for row in designs
        ],
    )

    preload_by_role: dict[str, dict[float, dict[str, str]]] = defaultdict(dict)
    for row in preloads:
        preload_by_role[row["role"]][float(row["preload_N"])] = row
    preload_table = md_table(
        [
            "构型",
            "成功率0.5/1/2 N",
            "起载中位0.5/1/2 N/mm",
            "净承载0.5/1/2 N",
        ],
        [
            [
                role,
                " / ".join(
                    pct(preload_by_role[role][value]["establish25_success_rate"])
                    for value in (0.5, 1.0, 2.0)
                ),
                " / ".join(
                    num(
                        preload_by_role[role][value][
                            "establish25_median_mm_successes"
                        ],
                        2,
                    )
                    for value in (0.5, 1.0, 2.0)
                ),
                " / ".join(
                    num(
                        preload_by_role[role][value][
                            "net_force_ratio_mean_median"
                        ]
                    )
                    for value in (0.5, 1.0, 2.0)
                ),
            ]
            for role in [f"优势候选A{index}" for index in range(1, 6)]
        ],
    )

    terrain_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in terrain_preloads:
        terrain_groups[row["terrain_stratum"]].append(row)
    terrain_table_rows: list[list[str]] = []
    for terrain in ("P40", "P60", "P100", "P180", "P240", "red_brick", "concrete"):
        rows = terrain_groups[terrain]
        case_total = sum(float(row["case_count"]) for row in rows)
        success = sum(
            float(row["case_count"]) * float(row["establish25_success_rate"])
            for row in rows
        ) / case_total
        terrain_table_rows.append(
            [
                terrain,
                pct(success),
                num(
                    sorted(
                        float(row["establish25_median_mm_successes"])
                        for row in rows
                    )[1],
                    2,
                ),
                num(
                    sorted(float(row["duty25_median"]) for row in rows)[1]
                ),
                num(
                    sorted(
                        float(row["net_force_ratio_mean_median"])
                        for row in rows
                    )[1]
                ),
            ]
        )
    terrain_table = md_table(
        ["地形", "稳定成功率", "起载中位/mm", "占空比", "净承载"],
        terrain_table_rows,
    )

    recommendation_table = md_table(
        ["层级", "构型", "用途", "判断"],
        [
            [
                row["recommendation_level"],
                row["role"],
                row["recommended_mechanism"],
                row["recommendation_reason"],
            ]
            for row in recommendation_rows
        ],
    )

    k_300_800_success = pair[("刚度300→800", "establish_success")]
    k_800_2000_success = pair[("刚度800→2000", "establish_success")]
    k_300_2000_net = pair[("刚度300→2000", "net_force_ratio_mean")]
    k_300_2000_neff = pair[
        ("刚度300→2000", "bearing_neff_fraction_median")
    ]
    orientation_success = pair[
        ("方向2×5→5×2", "establish_success")
    ]
    orientation_net = pair[
        ("方向2×5→5×2", "net_force_ratio_mean")
    ]
    rigid_success = pair[
        ("刚性快速A1→高承载A2", "establish_success")
    ]
    rigid_positive = pair[
        (
            "刚性快速A1→高承载A2",
            "positive_force_ratio_mean_clipped2",
        )
    ]
    rigid_net = pair[
        ("刚性快速A1→高承载A2", "net_force_ratio_mean")
    ]
    flexible_success = pair[
        ("柔顺300 A4→800 A5", "establish_success")
    ]
    flexible_net = pair[
        ("柔顺300 A4→800 A5", "net_force_ratio_mean")
    ]

    report = f"""# M3独立阵列12构型终筛分析报告

## 摘要

终筛已完成12个构型、300个地形条件、0.5/1/2 N三档恒定法向预载，共10,800个case。每个case拖拽10 mm、步长0.05 mm，共201个空间站，并保存全局与逐针完整路径。运行状态为`complete`，300/300个地形分片完成，耗时{float(source_manifest['elapsed_s']) / 60.0:.1f}分钟，错误日志为空。

终筛确认了三件关键事实：

1. **刚性阵列确实能够快速并持续建立抗拖承载。** A1在严格0.5 mm稳定窗口下成功率最高（{pct(design_by_id[PRIMARY_IDS[0]]['establish25_success_rate'])}），成功case起载距离中位数{num(design_by_id[PRIMARY_IDS[0]]['establish25_median_mm_successes'], 2)} mm；A2的稳健正向承载和净承载最高，形成清晰的“起载速度—承载幅值”双目标。
2. **柔顺不是单调收益。** A4实现最高的柔顺分载，但净承载为{num(design_by_id['m3_full_design_84bc8db4f9902f13b75c']['net_force_ratio_mean_median'])}P；A5牺牲一部分稳定起载，换取紧凑尺寸和略为正的净承载。
3. **匹配对照提供了可用于论文的因果证据。** 在完全相同几何下，300→2000 N/m使净承载平均改善{ci_text(k_300_2000_net)}，同时Neff/N变化为{ci_text(k_300_2000_neff)}；2×5和5×2方向差异则很小。

因此不建议把12个构型压缩成一个自动排名。建议保留4个主工程构型（A1、A2、A3、A5），并把A4、C1/C2、刚度三联和方向成对构型作为机理论文对照；原始12构型数据全部保留。

## 1. 数据与计算口径

- 仿真语义：恒定法向预载，拖动过程中失配后重新压合，并允许邻域换落点；
- 路径：10 mm，`dx=0.05 mm`；
- 地形：现有300个M1地形条件，没有复制或生成新地形；
- 正力方向：`Fx>0`表示阻碍`+x`拖动；
- 未接受站点：按零承载计入，不从时间/空间序列删除；
- 稳定起载主判据：第一个0.5 mm窗口内，11个站点至少10个满足`Fx/P≥0.25`；
- 正向累计承载：`mean(clip(Fx/P,0,2))`；
- 配对置信区间：先在同一地形ID内平均三档预载，再对300个地形ID进行成簇bootstrap。

### 1.1 0.05 mm步长带来的窗口修正

细筛的0.1 mm步长下，6个站点覆盖0.5 mm；终筛改为0.05 mm后，物理上相同的窗口必须改成11个站点。如果继续使用固定6站点，实际只检验0.25 mm，会显著高估“持续建立承载”的成功率。

{sensitivity_table}

多数优势候选在0.25 mm短窗口下接近100%，但在严格0.5 mm窗口下降到37%–69%。这不是仿真退化，而是说明**短时承载很常见，真正区分构型的是能否连续维持半毫米**。首次越过阈值的距离大多接近0，因此“搜索距离”主要由承载连续性决定，而不是第一次碰到凸起的距离。

## 2. 12构型总体结果

{design_table}

![12构型综合指标](../04_图表/终筛图1_12构型综合指标.png)

![搜索距离与承载权衡](../04_图表/终筛图2_搜索距离与承载权衡.png)

图2左上且圆点较大的构型代表短搜索、高累计承载和高成功率。A1、A2和A4构成主要Pareto区域；A3以无缺口和正净承载提供连续接触基线。刚度三联位于较低承载区，但它们的价值是严格匹配的机理证据，而不是作为工程赢家。

## 3. 刚性阵列：不是失败，而是重新压合驱动的单针承载

{gap_table}

A1、A2、C1和C2分别有78%–85%的case至少出现过一个未接受站点，但典型case只有约2.5%–4.0%的站点缺口。它们在约82%的站点触发再压合；A1、A2和C2还频繁使用邻域换落点。建立承载后，典型最长低载区只有0.20–0.40 mm，恢复距离约0.15–0.24 mm，路径末端低载中位数为零。

这组数据支持用户提出的工程过程：**局部崩开并不会终止拖拽，系统通过重新压合和邻域换落点继续承载。** 但刚性阵列的承载状态下`max_load_share=1`，主要由单针承担载荷；当前准静态模型没有针尖损伤和冲击，因此其寿命风险必须留到结构FEA或实验验证。

### 3.1 A1与A2的速度—强度权衡

以同一300个地形ID做配对：

- A2相对A1的稳定成功率变化为{ci_text(rigid_success)}，即A1明显更容易形成连续0.5 mm承载；
- A2相对A1的稳健正向累计承载提高{ci_text(rigid_positive)}；
- A2相对A1的净承载提高{ci_text(rigid_net)}；
- A1起载中位数{num(design_by_id[PRIMARY_IDS[0]]['establish25_median_mm_successes'], 2)} mm，A2为{num(design_by_id[PRIMARY_IDS[1]]['establish25_median_mm_successes'], 2)} mm。

A1应作为“最快稳定起载”构型；A2应作为“高承载”构型。两者不存在单一优劣，不能只留一个。

## 4. 高刚度连续承载与柔顺分载

### 4.1 A3：大阵列高刚度柔顺基线

A3为6×6、6 mm、固定80°、d0.6 mm、r100 μm、2000 N/m。它没有任何支撑缺口，净承载{num(design_by_id[PRIMARY_IDS[2]]['net_force_ratio_mean_median'])}P；承载时`Neff/N={num(design_by_id[PRIMARY_IDS[2]]['bearing_neff_fraction_median'])}`，约相当于5根针有效分担载荷，最大单针份额中位数约{pct(design_by_id[PRIMARY_IDS[2]]['bearing_max_load_share_median'])}。

其严格稳定成功率只有{pct(design_by_id[PRIMARY_IDS[2]]['establish25_success_rate'])}，且随预载提高而下降，说明“全程几何可达”不等于“连续达到0.25P”。它适合作为无缺口连续接触基线，而不是以起载成功率定义的唯一最优解。

### 4.2 A4与A5：稳定性、净承载和制造尺度的交换

{preload_table}

配对结果显示，A5相对A4：

- 稳定成功率变化{ci_text(flexible_success)}，A4更容易建立连续承载；
- 净承载提高{ci_text(flexible_net)}，A5总体方向更有利；
- 二者正向累计承载近似相同；
- A4和A5均有约3/4针参与有效分载。

A4保留为“低刚度、高均载但存在负向区段”的机理边界；A5保留为“尺寸更小、净承载略正”的工程柔顺候选。

![五个优势候选力—距曲线](../04_图表/终筛图4_五个优势候选力距曲线.png)

## 5. 严格匹配刚度三联

K300、K800、K2000均为2×2、6 mm、固定60°、d0.8 mm、r50 μm，仅刚度不同。

![严格匹配刚度三联](../04_图表/终筛图3_严格匹配刚度三联力距曲线.png)

配对结果为：

- 300→800 N/m的稳定成功率差为{ci_text(k_300_800_success)}，没有明确变化；
- 800→2000 N/m的稳定成功率提高{ci_text(k_800_2000_success)}；
- 300→2000 N/m的净承载平均提高{ci_text(k_300_2000_net)}；
- 300→2000 N/m的Neff/N变化为{ci_text(k_300_2000_neff)}，说明提高刚度伴随载荷共享下降。

因此该匹配几何下，2000 N/m明显提高切向力传递和净承载，但代价是更强的载荷集中。300与800 N/m的稳定起载差异不显著，说明从“软”提高到“中等”仍不足以跨过该固定60°、r50 μm参数包的主要几何限制。

## 6. 2×5与5×2方向对照

两者严格匹配，仅交换阵列长边方向。5×2相对2×5：

- 稳定成功率变化{ci_text(orientation_success)}；
- 净承载变化{ci_text(orientation_net)}；
- 正向累计承载几乎不变。

上述置信区间均覆盖零或效应很小。因此在5 mm、60→80°、d0.6 mm、r100 μm、2000 N/m参数包下，长边沿`+x`或横向布置不是强主效应。论文应将其报告为近似无差异的匹配结果，避免从非匹配构型推断普遍方向优势。

## 7. 地形和预载分层

{terrain_table}

P240几乎所有构型都能形成连续0.5 mm承载；P60和红砖的严格成功率最低。红砖仍有正净承载，说明它常出现“有承载但连续窗口不足”；P60在1 N和2 N下总体净承载转负，表明该地形尺度与当前角度/柔顺组合容易产生不利法向投影。

预载效应不是统一的：A1、A2和A4随预载提高，严格稳定成功率上升；A3则从0.5 N的{pct(preload_by_role['优势候选A3'][0.5]['establish25_success_rate'])}下降到2 N的{pct(preload_by_role['优势候选A3'][2.0]['establish25_success_rate'])}。因此预载不能作为独立比例系数外推，必须与阵列刚度和角度共同讨论。

## 8. 最终保留建议

{recommendation_table}

不建议删除任何终筛数据。推荐按以下层次使用：

1. **工程主构型4个：** A1快速起载、A2高承载、A3无缺口连续承载、A5紧凑柔顺分载；
2. **柔顺边界：** A4，用于解释均载收益与负向投影代价；
3. **刚性机制：** C1/C2，用于最小阵列和角度梯度讨论，其中C2不是严格单变量角度对照；
4. **严格刚度证据：** K300/K800/K2000三者必须成组保留；
5. **严格方向证据：** D2×5/D5×2必须成对保留。

这相当于“4个工程候选＋4组论文对照”，比机械地压缩为6个构型更适合两篇论文。若后续只能制造4套样机，优先A1、A2、A3、A5；若增加一套柔顺边界样机，再加入A4。

## 9. 两篇论文的使用建议

### 论文一：恒定预载下的承载建立与重新压合

主结果使用A1、A2、A3及刚度三联：

- A1/A2说明快速起载与高承载的双目标；
- 刚性case高缺口率与低站点缺口比例说明不能以“出现过缺口”判失败；
- 再压合、邻域换落点、低载区长度和恢复距离说明拖拽能继续；
- 刚度三联证明切向传递增强与分载下降同时发生。

### 论文二：阵列构型、角度和几何参数筛选

主结果使用A3、A4、A5、C1/C2和方向对照：

- A3/A4/A5比较阵列规模、刚度和制造尺度；
- C1说明2×2刚性阵列仍可工作；
- C2作为梯度机制构型，但不作严格角度因果结论；
- D2×5/D5×2提供方向弱效应的严格匹配证据。

## 10. 结论边界

1. 结果证明的是当前准静态降阶模型中的相对承载机理，不是实际墙面的绝对承载标定；
2. 红砖和混凝土地形仍待实测标定；
3. 刚性阵列的单针集中风险需要局部结构FEA、损伤模型或实验；
4. 10 μm地形分辨率足够本轮筛选；后续只需对A1/A2/A3/A5做少量5 μm同地形收敛验证，不需要重建全量地形库；
5. 10 mm路径足以评价起载、维持和短程恢复，但不代表磨损寿命。

## 11. 归档文件

- `01_原始数据/terminal_scan_12`：终筛原始摘要和完整路径；
- `02_派生数据/终筛/terminal_case_metrics.parquet`：10,800个case派生指标；
- `02_派生数据/终筛/terminal_design_metrics.csv`：12构型汇总；
- `02_派生数据/终筛/terminal_paired_comparisons.csv`：地形ID配对差值和95%置信区间；
- `02_派生数据/终筛/terminal_recommendations.csv`：最终分层建议；
- `04_图表/终筛图*.png`：终筛图件。

分析清单：`{analysis_manifest['schema_version']}`；主判据：`{analysis_manifest['establishment_rule']}`。
"""

    report_path = report_dir / "M3_12构型终筛分析报告_v1.md"
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "recommendations": str(
                    derived / "terminal_recommendations.csv"
                ),
                "primary_design_ids": list(PRIMARY_IDS),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
