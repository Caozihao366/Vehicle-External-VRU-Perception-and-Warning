# A2D2 六相机去畸变、YOLO 检测与 BEV

本工程已适配 A2D2 Sensor Fusion 数据集：自动读取六路相机图像，并直接使用
`cams_lidars.json` 中的相机内参、畸变参数和 `view` 外参。

处理流程：

`A2D2 六路 PNG -> 按 sample ID 同步 -> 去畸变 -> YOLO 检测 -> 地面 BEV -> 结果导出`

支持的数据目录既可以是 A2D2 根目录（其中包含 `camera_lidar`），也可以直接是
`camera_lidar` 目录。未指定 `--scene` 时使用找到的第一个序列。

## 运行

在 VS Code 中打开 `detect.py` 后直接点击“运行 Python 文件”即可。也可以在工程目录执行：

```powershell
python detect.py
```

默认配置为：

- 数据集：`D:\test2\a2d2-preview`
- 标定：`D:\test2\a2d2-preview\cams_lidars.json`
- 权重：`D:\test2\bev_new2\yolov8n.pt`
- 输出：工程目录下的 `output_a2d2`
- 处理帧数：1

如需覆盖默认值，可传入完整参数：

```powershell
python -m pip install -r requirements.txt
python detect.py `
  --input "D:\test2\a2d2-preview" `
  --calibration "D:\test2\a2d2-preview\cams_lidars.json" `
  --weights "yolov8n.pt" `
  --output ".\output" `
  --scene "20180810_150607" `
  --max-frames 1
```

当标定文件位于 `INPUT/cams_lidars.json` 时，可以省略 `--calibration`。

每帧输出六张去畸变检测图、`bev.jpg`，以及汇总检测框和车辆坐标的
`summary.json`。A2D2 原始图像会按官方参数校正：鱼眼相机使用 OpenCV fisheye
模型，前中长焦相机使用普通 pinhole 模型，输出内参采用 `CamMatrix`。
