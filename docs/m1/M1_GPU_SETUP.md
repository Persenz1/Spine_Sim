# M1 CUDA 虚拟环境

> 本页复现 2026-07-27 `defined_geometry` GPU 基线。当前
> `material_hybrid` 地形库生成只支持 CPU；不要用本页替代材料生成验收。

本机验证环境：

- NVIDIA GeForce RTX 4060 Ti，8 GB；
- NVIDIA 驱动 610.74，CUDA UMD 13.3；
- Python 3.13；
- CuPy 14.1.1；
- CUDA Toolkit wheel 13.3.1。

仓库内隔离环境：

```powershell
$python313 = "C:\path\to\python.exe"
& $python313 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e ".[test,gpu-cuda13]"
```

验证：

```powershell
.\.venv\Scripts\python.exe -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

生成 10 工况测试库：

```powershell
.\.venv\Scripts\spine-terrain.exe generate-suite `
  results/m1_gpu_suite/terrain_library `
  examples/m1_gpu_terrain_suite.json `
  --output results/m1_gpu_suite/suite_report.json `
  --tile-rows 64
```

`.venv/`、`results/` 和地形库二进制均由 `.gitignore` 排除。仓库只提交依赖声明、
工况配方、代码、测试和汇总报告。
