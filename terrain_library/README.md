# M1 本地地形库

`TerrainLibrary` 首次打开此目录时会建立：

```text
terrain_library/
├── recipes/
├── sources/
├── regions/
├── tracks/
├── manifests/
└── validation/
```

该目录是可重建缓存，不提交大型二进制文件。`recipes/` 和 `manifests/`
定义重建身份；`regions/` 保存 10 μm `float32 .npy` 原始高度；
`tracks/` 保存 M2/M3 使用的一维有限针尖几何；`sources/` 只登记文件高度图
的原始路径、单位、网格和哈希。

生产地形以 `np.load(path, mmap_mode="r")` 只读打开。删除区域缓存时保留
recipe 和 manifest，之后可用 `spine-terrain rebuild-region` 原样重建。
