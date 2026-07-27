# M1 轻量地形绘图

`spine-terrain plot-region` 从已经完成的 M1 地形库中只读打开
`raw_height.npy`，随后直接按显示分辨率抽取一个局部窗口。它不会载入整幅地形、
重新生成地形或修改地形缓存。

安装绘图可选依赖：

```powershell
python -m pip install -e ".[plot]"
```

生成三张 PNG 和一份 JSON 元数据：

```powershell
spine-terrain plot-region `
  results/m1_gpu_suite/terrain_library `
  terrain_recipe_c27494c6922e10a5f4a8 `
  region_3794b9e58fab7ff30845 `
  output `
  --center-x-mm 0 `
  --center-y-mm 0 `
  --overview-size-mm 10 `
  --sphere-radius-um 100 `
  --prefix baseline-terrain-10mm
```

输出包括：

- 10 × 10 mm 斜俯视 3D 地形图，坐标和色条注明单位，图内带水平标尺；
  垂直放大倍数不超过 1.5。
- 10 × 10 mm 2D 高度图，颜色表示高度，带高度色条和水平标尺。白色圆圈
  标出球尖局部图所在的沟槽。
- 自动选择沟槽附近 0.5 × 0.5 mm 局部地形，与 100 μm 有效球尖进行
  斜俯视 3D 对比。沟槽按“中心低、约 0.2 mm 周边较高”的规则从全景中选择；
  相机朝实际支撑点方向旋转，避免接触位置被前景地形遮住。
  球心高度取固定 x-y 位置下、对离散地形节点不发生穿透的最低值；红色菱形
  为控制球心高度的支撑点。该图采用正投影，x、y、z 三轴为相同物理比例，
  不放大 z 轴。

默认 10 mm 全场包含原始 1001 × 1001 节点，3D 曲面再降至最多
181 × 181 个显示节点。球尖落位
仍使用 0.5 mm 局部球足迹内的全部原始 10 μm 网格节点，因此显示降采样不会
改变球尖位置。
