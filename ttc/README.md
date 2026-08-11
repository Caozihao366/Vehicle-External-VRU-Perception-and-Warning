# TTC V1 轨迹碰撞风险模块

## 1. 模块定位

本目录提供车外VRU轨迹碰撞风险计算。模块在统一米制坐标和统一参考时刻下，
比较自车未来行驶包络与每个VRU的多模态预测轨迹，输出：

- 首次预测碰撞时间TTC；
- 预测时域内最小有符号净间距；
- 碰撞模态概率质量；
- 风险等级和主要触发原因；
- 每条预测模态的独立TTC结果；
- 可选的相机画面、BEV和离线视频可视化。

模块不负责目标检测、跟踪、像素转米制坐标、时间同步、轨迹模型训练、自车规划或
车辆制动执行。这些信息由上游感知、BEV、时间同步、轨迹预测和planner提供。

## 2. 目录结构

```text
ttc/
├─ __init__.py
├─ __main__.py
├─ models.py
├─ core.py
├─ risk.py
├─ adapter.py
├─ visualization.py
├─ integrations/
│  ├─ __init__.py
│  └─ qcnet.py
├─ requirements.txt
├─ requirements-qcnet-cu118.txt
└─ README.md
```

文件职责：

| 文件 | 职责 |
|---|---|
| `models.py` | 自车、VRU、轨迹模态和结果数据结构 |
| `core.py` | 轨迹生成、插值、几何净间距、碰撞搜索和多模态TTC |
| `risk.py` | 风险枚举、行为修正和连续帧风险状态机 |
| `adapter.py` | 通用JSON/字典接口、参考点转换和时间质量检查 |
| `integrations/qcnet.py` | 在同一Python进程中加载QCNet并转换为TTC轨迹模态 |
| `visualization.py` | 可选相机画面、米制BEV、多目标排序和MP4写出 |
| `__main__.py` | `python -m ttc`命令行入口 |

## 3. 正式运行环境

推荐统一环境：

| 组件 | 版本 |
|---|---|
| Python | 3.10.11 |
| PyTorch | 2.0.1+cu118 |
| CUDA运行时 | 11.8 |
| torch-geometric | 2.3.0 |
| torch-scatter | 2.1.1+pt20cu118 |
| torch-sparse | 0.6.17+pt20cu118 |
| torch-cluster | 1.6.1+pt20cu118 |
| NumPy | 1.26.4 |
| OpenCV Python | 4.10.0.84 |

TTC核心计算只依赖Python标准库。QCNet直连需要PyTorch和PyG；可视化需要NumPy和OpenCV。
没有使用QCNet或可视化时，不会主动导入这些可选依赖。

### 3.1 创建环境

```powershell
D:\shixi\Python310\python.exe -m venv D:\shixi\-\.venv310
D:\shixi\-\.venv310\Scripts\Activate.ps1
```

### 3.2 安装通用依赖

```powershell
python -m pip install -r ttc\requirements.txt
```

### 3.3 安装QCNet CUDA 11.8依赖

```powershell
python -m pip install -r ttc\requirements-qcnet-cu118.txt
```

虚拟环境目录不属于交付代码，不应放入TTC文件夹或上传。

## 4. 数据流

```text
检测与多目标跟踪
        ↓
稳定track_id和目标类别
        ↓
BEV/深度模块提供米制位置
        ↓
时间同步并维护5秒历史
        ↓
QCNet输出每目标6条未来2秒轨迹
        ↓
integrations/qcnet.py补t=0并转换为PredictionMode
        ↓
planner自车轨迹；缺失时使用CTRV
        ↓
逐目标多模态TTC
        ↓
风险状态机
        ↓
结构化报文 + 可选可视化
```

同一帧中的所有有效VRU应分别计算TTC，再统一排序。不要先选一个目标再计算。

## 5. 坐标系与单位

通用JSON接口必须声明：

```json
{"coordinate_frame": "ego_current"}
```

约定：

| 项目 | 定义 |
|---|---|
| 原点 | 当前参考时刻自车几何中心 |
| x轴 | 指向车前 |
| y轴 | 指向车左 |
| yaw | 从+x轴逆时针为正 |
| 距离 | 米 |
| 时间 | 输入为秒，TTC输出为毫秒 |
| 速度 | 米/秒 |
| 横摆率 | 弧度/秒 |
| 轨迹参考 | 所有未来点位于同一个当前自车坐标系 |

不能直接输入图像像素、经纬度、未转换世界坐标或不同参考时刻的轨迹。

轨迹参考点默认为车辆几何中心：

```json
{"reference_point": "geometric_center"}
```

如果planner输出后轴中心：

```json
{
  "reference_point": "rear_axle_center",
  "rear_axle_to_center_m": 1.45
}
```

通用适配器会转换到车辆几何中心。上游不要重复转换。

## 6. TTC公共接口

```python
from ttc import (
    EgoState,
    VRUState,
    EgoTrajectoryPoint,
    VRUTrajectoryPoint,
    PredictionMode,
    TTCResult,
    MultimodalTTCResult,
    RiskStateMachine,
    build_ego_trajectory,
    make_cv_prediction_mode,
    compute_ttc,
    compute_path_ttc,
    compute_multimodal_ttc,
    compute_multimodal_ttc_from_payload,
)
```

### 6.1 CTRV自车轨迹

```python
from ttc import EgoState, build_ego_trajectory

ego_trajectory = build_ego_trajectory(
    EgoState(
        x=0.0,
        y=0.0,
        yaw=0.0,
        speed_mps=5.0,
        yaw_rate_rps=0.0,
    ),
    horizon_s=2.0,
    dt_s=0.1,
)
```

如果planner已提供未来轨迹，应直接使用planner轨迹。CTRV只作为planner缺失时的回退。

### 6.2 多模态TTC

```python
from ttc import compute_multimodal_ttc

result = compute_multimodal_ttc(
    track_id=track_id,
    ego_trajectory=ego_trajectory,
    modes=prediction_modes,
    prediction_source="qcnet_ego_2s_6mode",
    behavior_state=behavior_state,
    sensor_availability=sensor_availability,
    ego_trajectory_source="local_planner",
)
```

每个目标单独调用一次。相同目标的多条轨迹作为`modes`一次传入。

## 7. QCNet直连接口

QCNet在同一个Python 3.10进程中加载，不需要启动另一个虚拟环境、子进程或中间文件。

```python
from ttc.integrations import QCNetTrajectoryProvider

with QCNetTrajectoryProvider(device="cuda") as provider:
    predictions = provider.predict(
        model_input,
        target_ids=["pedestrian_17", "bicycle_08"],
    )

pedestrian = predictions["pedestrian_17"]
prediction_modes = pedestrian.modes
```

`model_input`遵循轨迹模型部署包接口：

| 字段 | Shape/类型 | 单位 | 说明 |
|---|---|---|---|
| `position` | `[N,50,2]` | m | 固定t0自车坐标中的5秒历史位置 |
| `velocity` | `[N,50,2]` | m/s | 固定t0自车坐标中的历史速度 |
| `heading` | `[N,50]` | rad | 相对t0自车航向 |
| `valid_mask` | `[N,50]` | - | 历史点是否有效 |
| `agent_type` | `[N]` | - | 模型类别索引 |
| `av_index` | int | - | 自车在agent数组中的索引 |
| `agent_ids` | `list[str]` | - | 与track_id一致的稳定ID |
| `coordinate_frame` | `"ego"` | - | 固定t0自车坐标标签 |
| `frequency_hz` | `10.0` | Hz | 历史采样频率 |

模型对每个当前有效目标输出：

```text
trajectories:  [P,6,20,2]
probabilities: [P,6]
agent_indices: [P]
agent_ids:     P个稳定ID
```

未来点对应`t0+0.1s ... t0+2.0s`。TTC要求轨迹从`t=0`开始，
`QCNetTrajectoryProvider`会自动补入当前跟踪位置，最终每个模态包含21个点：

```text
t = 0.0, 0.1, 0.2, ... 2.0秒
```

每个目标返回：

```python
QCNetTargetPrediction(
    track_id=...,
    modes=(6个PredictionMode),
    inference_time_ms=...,
    prediction_source="qcnet_ego_2s_6mode",
    probability_semantics="model_softmax_relative_mode_probability",
    prediction_horizon_s=2.0,
)
```

模型概率是候选轨迹相对概率，不是碰撞概率。TTC逐模态做几何检查，再汇总碰撞模态概率。

## 8. 自车轨迹来源

优先级：

```text
planner未来轨迹
    ↓ 缺失
CTRV fallback
```

planner轨迹点格式：

```json
{"t": 0.1, "x": 0.5, "y": 0.0, "yaw": 0.0}
```

必须从`t=0`开始，时间严格递增，并与VRU轨迹位于相同参考时刻和坐标系。

QCNet可能同时输出自车预测，但该输出不是车辆可执行planner轨迹，不应默认替代planner。

## 9. 轨迹输入约束

每个`PredictionMode`包含：

| 字段 | 约束 |
|---|---|
| `mode_id` | 同一目标内唯一 |
| `probability` | 有限、非负；全部模态总和大于0 |
| `points` | 至少一个点，从`t=0`开始 |

每个`VRUTrajectoryPoint`包含：

| 字段 | 单位 | 说明 |
|---|---|---|
| `t_s` | s | 相对当前参考时刻 |
| `x` | m | 自车前向坐标 |
| `y` | m | 自车左向坐标 |
| `sigma_x_m` | m | 可选纵向标准差 |
| `sigma_y_m` | m | 可选横向标准差 |

时间戳必须严格递增，所有数值必须有限。不要把多个模态平均成一条轨迹。

当前QCNet输出不包含逐点标准差，因此直连模块生成的`sigma_x_m/sigma_y_m`为0。
TTC仍可使用基础不确定性膨胀参数。

## 10. TTC计算

### 10.1 自车占用

自车使用随未来位置和yaw移动的有向矩形：

```text
有效长度 = 车辆长度 + 2 × safety_margin
有效宽度 = 车辆宽度 + 2 × safety_margin
```

### 10.2 VRU占用

VRU使用圆形近似：

```text
有效半径(t)
= VRU物理半径
+ uncertainty_base
+ uncertainty_growth × t
+ sigma_scale × max(sigma_x, sigma_y)
```

这是保守占用近似，不是完整二维高斯或蒙特卡洛碰撞概率。

### 10.3 净间距与TTC

每个检查时刻计算VRU圆与自车有向矩形的有符号净间距：

- 大于0：包络分离；
- 等于0：包络接触；
- 小于0：包络相交。

轨迹点之间进行插值，默认以0.02秒细粒度检查。第一次出现包络相交的时刻是该模态TTC。

多模态汇总：

```text
earliest_ttc = 所有碰撞模态中的最早TTC
collision_probability = 所有碰撞模态归一化概率之和
```

`collision_probability`是离散碰撞模态概率质量，不是经过真实道路校准的连续碰撞概率。

## 11. 主要参数

| 参数 | 默认值 | 单位 | 说明 |
|---|---:|---|---|
| `dt_s` | 0.1 | s | 普通轨迹时间步 |
| `collision_check_dt_s` | 0.02 | s | 碰撞搜索步长 |
| `ego_length_m` | 4.8 | m | 自车长度 |
| `ego_width_m` | 2.0 | m | 自车宽度 |
| `safety_margin_m` | 0.4 | m | 自车安全余量 |
| `vru_radius_m` | 0.4 | m | VRU圆半径 |
| `uncertainty_base_m` | 0.15 | m | 固定膨胀 |
| `uncertainty_growth_mps` | 0.20 | m/s | 随时域增长膨胀 |
| `sigma_scale` | 1.0 | - | 模型标准差缩放 |
| `danger_ttc_s` | 1.2 | s | danger基础阈值 |
| `warning_ttc_s` | 2.5 | s | warning基础阈值 |

计算参数会改变风险结果，应由项目统一配置，不要在可视化中单独修改。

## 12. 风险等级

合法风险枚举：

```text
safe < notice < warning < danger
```

合法行为状态：

```text
normal / running / falling / wrong_way / uncertain
```

行为状态不会修改物理TTC，只影响风险阈值和触发原因。

连续视频中为每个`track_id`维护一个`RiskStateMachine`：

- 风险升级立即生效；
- 风险降低需要连续帧确认；
- danger额外保持若干帧；
- 目标永久离开后清除对应状态。

## 13. 输出字段

| 字段 | 说明 |
|---|---|
| `track_id` | 稳定目标ID |
| `prediction_source` | 轨迹来源 |
| `sensor_availability` | available/degraded/unavailable |
| `behavior_state` | 当前行为状态 |
| `evaluated_horizon_ms` | 共同可评估时域 |
| `earliest_ttc_ms` | 所有碰撞模态中最早TTC |
| `most_likely_mode_id` | 最高概率模态 |
| `most_likely_ttc_ms` | 最高概率模态TTC |
| `collision_probability` | 碰撞模态概率质量 |
| `risk_level` | safe/notice/warning/danger |
| `major_trigger_reason` | 主要风险原因 |
| `requires_aeb` | 给执行层的建议标志 |
| `status` | 计算和输入状态 |
| `ego_trajectory_source` | planner来源或ctrv_fallback |
| `input_quality` | valid/input_stale等 |
| `ttc_resolution_ms` | 碰撞搜索分辨率 |
| `critical_mode_id` | 最早碰撞模态 |
| `critical_mode_probability` | 关键模态概率 |
| `min_clearance_m` | 最小有符号净间距 |
| `time_to_min_clearance_ms` | 到最小间距的时间 |
| `mode_results` | 各模态独立结果 |

`NO_COLLISION_WITHIN_HORIZON`只表示预测时域内未相交，不表示长期绝对安全。
`requires_aeb`不能直接连接车辆制动执行器。

## 14. 时间质量

实时输入建议提供：

```json
{
  "timing": {
    "source_timestamp_ms": 10000,
    "evaluation_timestamp_ms": 10080,
    "max_input_age_ms": 250
  }
}
```

输入过期时仍返回几何结果，但标记`input_stale`，不包装成可信安全结论。
TTC不自动猜测上游延迟，也不擅自平移预测轨迹。

## 15. 可选可视化

`visualization.py`只显示已经计算好的结果，不执行轨迹预测或TTC计算。

输入：

- 当前相机BGR帧；
- planner或CTRV自车轨迹；
- 每个VRU的模型预测模态；
- 每个VRU的TTC结果；
- 感知/匹配模块提供的可选相机像素点。

显示：

- 左侧相机画面和右侧米制BEV；
- 自车未来中心线和矩形包络；
- 全部VRU模型预测轨迹；
- 关键模态、碰撞点和可用的不确定性；
- 多目标风险排序和最高风险白环；
- TTC、净间距和碰撞模态概率；
- 自车轨迹来源、帧号和时间戳。

单帧：

```python
from ttc.visualization import (
    VisualizationConfig,
    VisualizationTarget,
    render_ttc_frame,
)

targets = [
    VisualizationTarget(
        track_id=track_id,
        category=category,
        modes=prediction_modes,
        result=ttc_result,
        image_point=(u, v),
    )
]

canvas = render_ttc_frame(
    camera_frame,
    ego_trajectory,
    targets,
    config=VisualizationConfig(title="VRU TTC"),
    frame_index=frame_index,
    timestamp_ms=timestamp_ms,
)
```

连续视频：

```python
from ttc.visualization import TTCVideoWriter

with TTCVideoWriter("output/ttc_demo.mp4", fps=source_fps) as writer:
    for frame in frames:
        canvas = render_ttc_frame(
            frame.camera_image,
            frame.ego_trajectory,
            frame.targets,
            frame_index=frame.index,
            timestamp_ms=frame.timestamp_ms,
        )
        writer.write(canvas)
```

可视化中的车辆尺寸和安全余量应与实际TTC调用保持一致。

## 16. 已知限制

- CTRV不是完整车辆规划器；
- VRU圆形包络是简化形状；
- 不确定性采用保守膨胀近似；
- QCNet当前不输出逐点标准差；
- 模型预测误差会直接传递到TTC；
- 标定、米制定位和时间同步误差会直接影响结果；
- 预测时域外风险无法由本次计算判断；
- 风险阈值需要结合实际项目场景标定；
- 当前V1是基准风险模块，不代表量产车辆安全认证。

## 17. 独立性与依赖

TTC代码为项目内独立实现。公开项目和论文仅用于核对轨迹碰撞风险的通用算法思想，
未复制CommonRoad-CriMe或Collision-Probability-Estimation源码，也未将其作为运行依赖。

QCNet模型代码和权重遵循模型目录中的许可证与来源说明。PyTorch、PyG、NumPy、OpenCV
适用各自许可证。
