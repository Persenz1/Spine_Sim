# M1 CUDA 虚拟环境

> 本页同时覆盖 2026-07-27 `defined_geometry` GPU 基线和
> 2026-07-29 `material-terrain-v2` 材料生成后端。

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

生成单块 CUDA 材料地形：

```powershell
.\.venv\Scripts\spine-terrain.exe generate-material output\P100.npz `
  --material sandpaper --subtype P100 `
  --size-x-mm 147.960 --size-y-mm 40.200 --resolution-um 10 `
  --seed 41003 --mode synthetic --backend cuda
```

生成用于 M3 改进测试的全尺寸 15 条件 catalog：

```powershell
.\.venv\Scripts\python.exe scripts\generate_m1_material_m3_test_batch.py `
  --output results\m1_material_m3_test --backend cuda
```

材料 CUDA 后端在 Y 方向分块计算相关场，并在 GPU 上顺序 stamping
孔洞、骨料和突出磨粒；实测砂纸 patch quilting 仍在 CPU。输出 metadata
记录 requested/resolved backend、CuPy/CUDA 版本、设备名和显存池峰值。

`.venv/`、`results/` 和地形库二进制均由 `.gitignore` 排除。仓库只提交依赖声明、
工况配方、代码、测试和汇总报告。
