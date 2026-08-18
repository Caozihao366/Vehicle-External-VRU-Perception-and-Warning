# FR05 逆行检测模块

可独立拷贝到总线工程中的子模块。规则：**同本车道 + 车头(front) → wrong_way**。

## 目录结构

```
fr05_wrongway/
├── __init__.py              # 导出 WrongWayPipeline
├── pipeline.py              # ★ 总线接入主入口 process_frame()
├── config.py                # 默认参数与权重路径
├── device_utils.py          # GPU 自动选择 (cuda > mps > cpu)
├── vru_tracker.py           # ByteTrack 后处理 + 卡尔曼滤波
├── wrong_way.py             # 逆行融合判定
├── vehicle_orientation.py   # ResNet18 front/rear
├── yolopv2_lane.py          # YOLOPv2 车道 + 本车道
├── demo.py                  # 本地视频 demo
├── requirements.txt
├── weights/                 # ★ 需放置权重（见下）
└── third_party/YOLOPv2/     # YOLOPv2 utils + 权重
```

## 权重准备

从原 Wrongway 项目复制到本包对应路径：

| 文件 | 目标路径 |
|---|---|
| `yolov8s.pt` | `fr05_wrongway/weights/yolov8s.pt` |
| `weights/vehicle_orientation.pt` | `fr05_wrongway/weights/vehicle_orientation.pt` |
| `YOLOPv2/data/weights/yolopv2.pt` | `fr05_wrongway/third_party/YOLOPv2/data/weights/yolopv2.pt` |

## 总线接入（推荐）

```python
from fr05_wrongway import WrongWayPipeline

pipe = WrongWayPipeline(device="auto")  # 或 mps / cuda:0

while True:
    frame = bus.read_camera_frame()  # BGR numpy (H,W,3)
    out = pipe.process_frame(frame)

    for obj in out["objects"]:
        print(obj["track_id"], obj["behavior"], obj["ego_relation"])

    for ev in out["wrong_way_events"]:
        bus.publish_alarm("wrong_way", ev)
```

### `process_frame` 返回字段

| 字段 | 说明 |
|---|---|
| `frame_idx` | 帧序号 |
| `fps` | 估计帧率 |
| `objects` | 全部目标列表（见下） |
| `wrong_way_events` | `behavior=="wrong_way"` 的子集 |
| `wrong_way_count` | 逆行目标数 |
| `ego_lane_source` | 本车道估计来源 |
| `lane_heading_deg` | 道路朝向（可选） |
| `annotated_frame` | `annotate=True` 时带车道叠加的 BGR 图 |

每个 `objects[i]`：

```python
{
    "track_id": 3,
    "class_id": 2,
    "class_name": "car",
    "bbox": [x1, y1, x2, y2],
    "bottom_center": [bx, by],
    "behavior": "normal" | "suspect_wrong_way" | "wrong_way",
    "behavior_conf": 0.0,
    "orient_face": "front" | "rear" | "unknown",
    "orient_conf": 0.91,
    "ego_relation": "in_ego" | "outside" | "invade_ego" | "unknown",
    "decision_source": "same_lane+front" | "front_but_other_lane" | "none",
}
```

## 本地 Demo

在 **Wrongway 仓库根目录** 执行：

```bash
./.venv/bin/python -m fr05_wrongway.demo --source videos/bli.mp4 --device auto
```

## 配置

修改 `fr05_wrongway/config.py`，或在创建 Pipeline 时传参：

```python
WrongWayPipeline(device="cuda:0", conf=0.35, enable_yolop=True)
```

## 依赖

```bash
pip install -r fr05_wrongway/requirements.txt
```

Ultralytics 会自动下载 ByteTrack 配置；首次运行 YOLOv8 若无权重需自行放置或让 ultralytics 拉取。
