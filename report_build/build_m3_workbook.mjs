import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const archiveRoot = process.argv[2];
const previewDir = process.argv[3];
if (!archiveRoot || !previewDir) {
  throw new Error("usage: node build_m3_workbook.mjs <archive-root> <preview-dir>");
}

const derivedDir = path.join(archiveRoot, "02_派生数据");
const reportDir = path.join(archiveRoot, "03_报告");
await fs.mkdir(reportDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const workbook = Workbook.create();

const imports = [
  ["Design Metrics", "design_bearing_metrics.csv"],
  ["Factor Metrics", "factor_bearing_metrics.csv"],
  ["Terminal Plan", "终筛12构型方案.csv"],
  ["Terrain Preload", "terrain_preload_bearing_metrics.csv"],
  ["Force Curves", "force_distance_curves_by_spring_preload.csv"],
  ["Terminal Results", "终筛/terminal_design_metrics.csv"],
  ["Terminal Pairs", "终筛/terminal_paired_comparisons.csv"],
  ["Terminal Preload", "终筛/terminal_design_preload_metrics.csv"],
  ["Terminal Terrain", "终筛/terminal_terrain_preload_metrics.csv"],
  ["Terminal Recs", "终筛/terminal_recommendations.csv"],
];

for (const [sheetName, filename] of imports) {
  const csvText = (
    await fs.readFile(path.join(derivedDir, filename), "utf8")
  ).replace(/^\uFEFF/, "");
  await workbook.fromCSV(csvText, { sheetName });
}

const overview = workbook.worksheets.add("Overview");
overview.showGridLines = false;
overview.getRange("A1:H1").merge();
overview.getRange("A1").values = [["M3独立阵列细筛：承载建立与构型筛选"]];
overview.getRange("A1:H1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 18 },
  horizontalAlignment: "left",
  verticalAlignment: "center",
  rowHeight: 32,
};
overview.getRange("A3:H3").merge();
overview.getRange("A3").values = [[
  "主判据：第一个0.5 mm窗口内至少5/6站点满足 Fx/preload ≥ 0.25；未接受站点按零承载处理。",
]];
overview.getRange("A3:H3").format = {
  fill: "#EAF0F8",
  font: { color: "#17365D", italic: true },
  wrapText: true,
  rowHeight: 32,
};

const factorSheet = workbook.worksheets.getItem("Factor Metrics");
const factorUsed = factorSheet.getUsedRange();
const factorValues = factorUsed.values;
const factorHeaders = factorValues[0].map((value) => String(value));
const factorHeaderIndex = new Map(
  factorHeaders.map((value, index) => [value, index]),
);
const colLetter = (index) => {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
};
const springFactorRows = [];
for (let row = 1; row < factorValues.length; row += 1) {
  if (String(factorValues[row][factorHeaderIndex.get("scope")]) === "all_by_spring") {
    springFactorRows.push(row + 1);
  }
}
const springOrder = new Map([["300", 0], ["800", 1], ["2000", 2], ["rigid", 3]]);
springFactorRows.sort((a, b) => {
  const av = String(factorValues[a - 1][factorHeaderIndex.get("spring_label")]);
  const bv = String(factorValues[b - 1][factorHeaderIndex.get("spring_label")]);
  return springOrder.get(av) - springOrder.get(bv);
});

const overviewHeaders = [
  "轴向支撑",
  "稳定起载成功率",
  "起载距离中位数/mm",
  "承载占空比",
  "稳健正向承载/Fx÷P",
  "净承载/Fx÷P",
  "有缺口case",
  "典型缺口站点",
];
overview.getRange("A5:H5").values = [overviewHeaders];
const overviewFields = [
  "spring_label",
  "establish25_success_rate",
  "establish25_median_mm_successes",
  "duty25_median",
  "positive_force_ratio_mean_clipped2_median",
  "net_force_ratio_mean_median",
  "case_any_gap_rate",
  "station_gap_fraction_median",
];
for (let outputRow = 6; outputRow <= 9; outputRow += 1) {
  const factorRow = springFactorRows[outputRow - 6];
  overview.getRange(`A${outputRow}:H${outputRow}`).formulas = [[
    ...overviewFields.map((field) => {
      const column = colLetter(factorHeaderIndex.get(field));
      return `='Factor Metrics'!${column}${factorRow}`;
    }),
  ]];
}
overview.getRange("A5:H9").format = {
  borders: { preset: "insideHorizontal", style: "thin", color: "#D6DEE8" },
};
overview.getRange("A5:H5").format = {
  fill: "#4472C4",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  rowHeight: 30,
};
overview.getRange("A6:A9").format.font = { bold: true };
overview.getRange("B6:B9").format.numberFormat = "0.0%";
overview.getRange("C6:C9").format.numberFormat = "0.00";
overview.getRange("D6:D9").format.numberFormat = "0.0%";
overview.getRange("E6:F9").format.numberFormat = "0.000";
overview.getRange("G6:H9").format.numberFormat = "0.0%";
overview.getRange("A5:H9").conditionalFormats.add("colorScale", {
  colors: ["#FCE8E6", "#FFF4CC", "#E6F4EA"],
  thresholds: ["min", "50%", "max"],
});

overview.getRange("A11:H11").merge();
overview.getRange("A11").values = [[
  "关键读取：刚性阵列有缺口case多，但典型缺口站点约2%；其稳定起载更快、累计正向承载更高。"
]];
overview.getRange("A11:H11").format = {
  fill: "#FFF2CC",
  font: { bold: true, color: "#7A4E00" },
  wrapText: true,
  rowHeight: 30,
};

overview.getRange("A13:H13").values = [[
  "输出索引", "用途", "原始/派生", "记录数", "主要单位", "统计边界", "论文建议", "备注",
]];
overview.getRange("A14:H18").values = [
  ["Design Metrics", "96构型比较", "派生", 96, "mm, N, mJ, ratio", "每构型450 case", "两篇共用", "不等同最终排名"],
  ["Factor Metrics", "参数主效应与交互", "派生", factorValues.length - 1, "ratio, mm", "含全体/柔顺分层", "构型论文", "边际趋势需匹配对照"],
  ["Terminal Plan", "12构型终筛方案", "派生", 12, "同Design Metrics", "机理平衡", "两篇共用", "终筛结论见Terminal Overview"],
  ["Terrain Preload", "地形×预载×弹簧", "派生", 84, "N, mm, ratio", "分层汇总", "承载论文", "红砖/混凝土未标定"],
  ["Force Curves", "弹簧×预载力—距离", "派生", 1212, "mm, N, ratio", "中位数与分位数", "承载论文", "完整case路径在Parquet/NPZ"],
];
overview.getRange("A13:H13").format = {
  fill: "#5B9BD5",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  wrapText: true,
};
overview.getRange("A13:H18").format.borders = {
  preset: "insideHorizontal",
  style: "thin",
  color: "#D6DEE8",
};
overview.getRange("D14:D18").format.numberFormat = "#,##0";

const chart1 = overview.charts.add("bar", overview.getRange("A5:B9"));
chart1.title = "稳定起载成功率";
chart1.hasLegend = false;
chart1.yAxis = { numberFormatCode: "0%" };
chart1.setPosition("J2", "P16");
overview.getRange("J34:K38").values = [
  ["spring_type", "median_establish_x_mm"],
  ["=A6", "=C6"],
  ["=A7", "=C7"],
  ["=A8", "=C8"],
  ["=A9", "=C9"],
];
const chart2 = overview.charts.add("bar", overview.getRange("J34:K38"));
chart2.title = "稳定起载距离（成功case中位数）";
chart2.hasLegend = false;
chart2.yAxis = { numberFormatCode: "0.0" };
chart2.setPosition("Q2", "W16");
overview.freezePanes.freezeRows(5);
overview.getRange("A1:H18").format.font.name = "Microsoft YaHei";
overview.getRange("A1:A18").format.columnWidth = 18;
overview.getRange("B1:B18").format.columnWidth = 24;
overview.getRange("C1:H18").format.columnWidth = 18;
overview.getRange("A13:H18").format.wrapText = true;

const terminalOverview = workbook.worksheets.add("Terminal Overview");
terminalOverview.showGridLines = false;
terminalOverview.getRange("A1:H1").merge();
terminalOverview.getRange("A1").values = [["M3独立阵列终筛：12构型承载建立与机理结论"]];
terminalOverview.getRange("A1:H1").format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF", size: 18 },
  horizontalAlignment: "left",
  verticalAlignment: "center",
  rowHeight: 32,
};
terminalOverview.getRange("A3:H3").merge();
terminalOverview.getRange("A3").values = [[
  "主判据：第一个0.5 mm窗口内至少10/11站点满足 Fx/preload ≥ 0.25；未接受站点按零承载处理。",
]];
terminalOverview.getRange("A3:H3").format = {
  fill: "#EAF0F8",
  font: { color: "#17365D", italic: true },
  wrapText: true,
  rowHeight: 32,
};

const terminalResultsSheet = workbook.worksheets.getItem("Terminal Results");
const terminalResultValues = terminalResultsSheet.getUsedRange().values;
const terminalResultHeaders = terminalResultValues[0].map((value) => String(value));
const terminalResultIndex = new Map(
  terminalResultHeaders.map((value, index) => [value, index]),
);
const terminalOverviewHeaders = [
  "构型",
  "稳定成功率",
  "起载中位/mm",
  "承载占空比",
  "正向承载/Fx÷P",
  "净承载/Fx÷P",
  "Neff/N",
  "有缺口case",
];
terminalOverview.getRange("A5:H5").values = [terminalOverviewHeaders];
const terminalOverviewFields = [
  "role",
  "establish25_success_rate",
  "establish25_median_mm_successes",
  "duty25_median",
  "positive_force_ratio_mean_clipped2_median",
  "net_force_ratio_mean_median",
  "bearing_neff_fraction_median",
  "case_any_gap_rate",
];
for (let outputRow = 6; outputRow <= 17; outputRow += 1) {
  const sourceRow = outputRow - 4;
  terminalOverview.getRange(`A${outputRow}:H${outputRow}`).formulas = [[
    ...terminalOverviewFields.map((field) => {
      const column = colLetter(terminalResultIndex.get(field));
      return `='Terminal Results'!${column}${sourceRow}`;
    }),
  ]];
}
terminalOverview.getRange("A5:H5").format = {
  fill: "#4472C4",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  rowHeight: 30,
};
terminalOverview.getRange("A5:H17").format.borders = {
  preset: "insideHorizontal",
  style: "thin",
  color: "#D6DEE8",
};
terminalOverview.getRange("A6:A17").format.font = { bold: true };
terminalOverview.getRange("B6:B17").format.numberFormat = "0.0%";
terminalOverview.getRange("C6:C17").format.numberFormat = "0.00";
terminalOverview.getRange("D6:D17").format.numberFormat = "0.0%";
terminalOverview.getRange("E6:G17").format.numberFormat = "0.000";
terminalOverview.getRange("H6:H17").format.numberFormat = "0.0%";
terminalOverview.getRange("B6:B17").conditionalFormats.add("colorScale", {
  colors: ["#FCE8E6", "#FFF4CC", "#E6F4EA"],
  thresholds: ["min", "50%", "max"],
});
terminalOverview.getRange("C6:C17").conditionalFormats.add("colorScale", {
  colors: ["#E6F4EA", "#FFF4CC", "#FCE8E6"],
  thresholds: ["min", "50%", "max"],
});
terminalOverview.getRange("F6:F17").conditionalFormats.add("colorScale", {
  colors: ["#FCE8E6", "#FFF4CC", "#E6F4EA"],
  thresholds: ["min", "50%", "max"],
});
terminalOverview.getRange("A19:H19").merge();
terminalOverview.getRange("A19").values = [[
  "结论：A1优先快速稳定起载，A2优先高净承载，A3提供无缺口连续承载，A5提供紧凑柔顺分载；A4和匹配对照保留用于论文机理。",
]];
terminalOverview.getRange("A19:H19").format = {
  fill: "#FFF2CC",
  font: { bold: true, color: "#7A4E00" },
  wrapText: true,
  rowHeight: 34,
};
terminalOverview.getRange("A21:D21").values = [[
  "主工程构型", "主要用途", "结构", "关键取舍",
]];
terminalOverview.getRange("A22:D25").values = [
  ["A1", "快速稳定起载", "4×4 / 5 mm / 70° / 刚性", "成功率最高；单针集中"],
  ["A2", "高净承载", "5×2 / 5 mm / 60° / 刚性", "承载最高；起载慢于A1"],
  ["A3", "无缺口连续承载", "6×6 / 6 mm / 80° / 2000", "约5针有效分载；尺寸最大"],
  ["A5", "紧凑柔顺分载", "2×2 / 6 mm / 60→80° / 800", "约3针分载；连续窗口成功率较低"],
];
terminalOverview.getRange("A21:D21").format = {
  fill: "#5B9BD5",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
};
terminalOverview.getRange("A21:D25").format.borders = {
  preset: "insideHorizontal",
  style: "thin",
  color: "#D6DEE8",
};
terminalOverview.getRange("A21:D25").format.wrapText = true;
terminalOverview.getRange("A22:D25").format.rowHeight = 44;
terminalOverview.getRange("J34:K46").values = [
  ["role", "success_rate"],
  ...Array.from({ length: 12 }, (_, index) => [
    `=RIGHT(A${index + 6},2)`,
    `=B${index + 6}`,
  ]),
];
const terminalChart1 = terminalOverview.charts.add(
  "bar",
  terminalOverview.getRange("J34:K46"),
);
terminalChart1.title = "严格0.5 mm稳定起载成功率";
terminalChart1.hasLegend = false;
terminalChart1.yAxis = { numberFormatCode: "0%" };
terminalChart1.setPosition("J2", "P17");
terminalOverview.getRange("M34:N46").values = [
  ["role", "net_force_ratio"],
  ...Array.from({ length: 12 }, (_, index) => [
    `=RIGHT(A${index + 6},2)`,
    `=F${index + 6}`,
  ]),
];
const terminalChart2 = terminalOverview.charts.add(
  "bar",
  terminalOverview.getRange("M34:N46"),
);
terminalChart2.title = "净承载（Fx/P）";
terminalChart2.hasLegend = false;
terminalChart2.yAxis = { numberFormatCode: "0.00" };
terminalChart2.setPosition("Q2", "W17");
terminalOverview.freezePanes.freezeRows(5);
terminalOverview.getRange("A1:H25").format.font.name = "Microsoft YaHei";
terminalOverview.getRange("A1:A25").format.columnWidth = 18;
terminalOverview.getRange("B1:B25").format.columnWidth = 20;
terminalOverview.getRange("C1:H25").format.columnWidth = 18;

const percentageHeaderRegex = /(rate|fraction|duty)/i;
const numericHeaderRegex = /(mm|ratio|share|work|force|preload|count|rank|stiffness|radius|diameter|spacing|angle|nx|ny)/i;
const tableNames = new Map([
  ["Design Metrics", "DesignMetricsTable"],
  ["Factor Metrics", "FactorMetricsTable"],
  ["Terminal Plan", "TerminalPlanTable"],
  ["Terrain Preload", "TerrainPreloadTable"],
  ["Force Curves", "ForceCurvesTable"],
  ["Terminal Results", "TerminalResultsTable"],
  ["Terminal Pairs", "TerminalPairsTable"],
  ["Terminal Preload", "TerminalPreloadTable"],
  ["Terminal Terrain", "TerminalTerrainTable"],
  ["Terminal Recs", "TerminalRecsTable"],
]);

for (const [sheetName] of imports) {
  const sheet = workbook.worksheets.getItem(sheetName);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const used = sheet.getUsedRange();
  const values = used.values;
  const headers = values[0].map((value) => String(value));
  used.format.font.name = "Microsoft YaHei";
  sheet.getRangeByIndexes(0, 0, 1, headers.length).format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    horizontalAlignment: "center",
    verticalAlignment: "center",
    rowHeight: 32,
  };
  for (let column = 0; column < headers.length; column += 1) {
    const header = headers[column];
    const range = sheet.getRangeByIndexes(1, column, Math.max(1, values.length - 1), 1);
    if (percentageHeaderRegex.test(header) && !/count/i.test(header)) {
      range.format.numberFormat = "0.0%";
    } else if (numericHeaderRegex.test(header)) {
      range.format.numberFormat = "0.000";
    }
    const lower = header.toLowerCase();
    let width = 14;
    if (lower.includes("design_id") || lower === "design_a" || lower === "design_b") width = 37;
    else if (lower.includes("reason") || lower.includes("definition")) width = 44;
    else if (lower.includes("mechanism") || lower.includes("scope")) width = 24;
    else if (lower.includes("pair_name") || lower.includes("metric_label")) width = 24;
    else if (lower.includes("comparison_kind") || lower.includes("recommendation_level")) width = 20;
    else if (lower.includes("terrain") || lower.includes("angle")) width = 18;
    else if (lower.includes("role")) width = 16;
    sheet.getRangeByIndexes(0, column, values.length, 1).format.columnWidth = width;
  }
  used.format.borders = {
    insideHorizontal: { style: "thin", color: "#E4E9EF" },
  };
  const tableRange = used.address;
  const table = sheet.tables.add(tableRange, true, tableNames.get(sheetName));
  table.style = "TableStyleMedium2";
  table.showBandedColumns = false;
  table.showFilterButton = true;
}

const designSheet = workbook.worksheets.getItem("Design Metrics");
const designHeaders = designSheet.getUsedRange().values[0].map((value) => String(value));
for (const field of [
  "establish25_success_rate",
  "duty25_median",
  "positive_force_ratio_mean_clipped2_median",
  "net_force_ratio_mean_median",
]) {
  const columnIndex = designHeaders.indexOf(field);
  if (columnIndex >= 0) {
    const range = designSheet.getRangeByIndexes(1, columnIndex, 96, 1);
    range.conditionalFormats.add("colorScale", {
      colors: ["#FCE8E6", "#FFF4CC", "#E6F4EA"],
      thresholds: ["min", "50%", "max"],
    });
  }
}

const terminalSheet = workbook.worksheets.getItem("Terminal Plan");
terminalSheet.getRange("A1:T13").format.wrapText = true;
terminalSheet.getRange("A2:B13").format.font = { bold: true, color: "#17365D" };
const terminalResults = workbook.worksheets.getItem("Terminal Results");
const terminalResultsHeaders = terminalResults.getUsedRange().values[0].map(
  (value) => String(value),
);
for (const field of [
  "establish25_success_rate",
  "duty25_median",
  "positive_force_ratio_mean_clipped2_median",
  "net_force_ratio_mean_median",
]) {
  const columnIndex = terminalResultsHeaders.indexOf(field);
  if (columnIndex >= 0) {
    terminalResults
      .getRangeByIndexes(1, columnIndex, 12, 1)
      .conditionalFormats.add("colorScale", {
        colors: ["#FCE8E6", "#FFF4CC", "#E6F4EA"],
        thresholds: ["min", "50%", "max"],
      });
  }
}

const inspectOverview = await workbook.inspect({
  kind: "table",
  range: "Overview!A1:H18",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 10,
});
await fs.writeFile(
  path.join(previewDir, "overview_inspect.ndjson"),
  inspectOverview.ndjson,
  "utf8",
);
const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(
  path.join(previewDir, "formula_errors.ndjson"),
  errorScan.ndjson,
  "utf8",
);

for (const [sheetName, range] of [
  ["Overview", "A1:W31"],
  ["Terminal Overview", "A1:W31"],
  ["Design Metrics", "A1:Z18"],
  ["Factor Metrics", "A1:Z18"],
  ["Terminal Plan", "A1:T13"],
  ["Terrain Preload", "A1:Z18"],
  ["Force Curves", "A1:R18"],
  ["Terminal Results", "A1:Z13"],
  ["Terminal Pairs", "A1:R18"],
  ["Terminal Preload", "A1:Z18"],
  ["Terminal Terrain", "A1:Z18"],
  ["Terminal Recs", "A1:T13"],
]) {
  const preview = await workbook.render({
    sheetName,
    range,
    scale: 1,
    format: "png",
  });
  const safeName = sheetName.replaceAll(" ", "_");
  await fs.writeFile(
    path.join(previewDir, `${safeName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(reportDir, "M3_细筛论文分析数据_v1.xlsx"));
