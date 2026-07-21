# 多相机 BEV 地面投影

本项目把 `input/` 中 nuScenes 风格的六路相机图像同步后，利用
`calibrated_sensor.json` 的内参与相机到车体外参，将地面（`z=0`）投影到鸟瞰图，
并在相机重叠区域做视角与边缘加权融合。

## 快速运行

```powershell
python -m pip install -r requirements.txt
python bev.py --max-frames 1
```

结果保存在 `output/<场景名>/`。批量处理首个场景：

```powershell
python bev.py
```

指定场景、帧和 BEV 范围：

```powershell
python bev.py `
  --scene "n008-2018-08-01-15-16-36-0400" `
  --frame 10 `
  --x-range -10 30 `
  --y-range -15 15 `
  --resolution 0.05
```

坐标采用车辆坐标系：`x` 向前、`y` 向左、`z` 向上；输出图像上方是车辆前方。
默认覆盖车后 10 m 至车前 30 m、左右各 15 m，分辨率 5 cm/像素。

## 参数

- `--scene`：文件名中第一个 `__` 前的场景名；不指定时处理时间最早的场景。
- `--frame`：只生成指定的零基帧编号，便于快速预览。
- `--max-frames`：限制输出帧数。
- `--tolerance-ms`：六路相机最近时间戳允许的最大偏差，默认 60 ms。
- `--x-range` / `--y-range`：BEV 覆盖范围（米）。
- `--resolution`：每像素代表的米数。

这是地面平面 IPM（Inverse Perspective Mapping），道路纹理效果最好；车辆、行人等
高于地面的物体会因平面假设产生拉伸，这是纯几何 BEV 的正常现象。若要得到语义 BEV
或消除动态物体拉伸，需要额外加入深度估计或 3D 感知模型。
