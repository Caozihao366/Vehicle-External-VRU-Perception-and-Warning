# BEV 组 × QCNet Ego 轨迹预测部署接口交接文档

## 1. 交接范围

本文用于 BEV 感知/跟踪组与轨迹预测组联合集成当前部署版本。

交付模型是 **Ego-coordinate、map-free、2 秒 QCNet**，不是 Trajectron++。目录名称
`trajectronpp_ego_bev` 仅沿用现有上层工程命名。

当前版本能力：

- BEV 输入频率：10 Hz。
- 历史窗口：50 帧，即 5 秒。
- 输入对象：AV 加多个动态目标。
- 地图输入：无。
- 模型：启用 Ego-relative 分支。
- 每个目标输出：6 条候选轨迹。
- 每条轨迹：20 个点，即未来 2 秒、10 Hz。
- 输入输出坐标：同一个固定预测时刻 `t0` Ego frame。
- 接口形态：实时 Python 对象调用，无 JSON 中间文件。

## 2. 双方职责边界

### BEV/跟踪组负责

1. 以 10 Hz 提供 `BEVFrame`。
2. 每个物理目标提供稳定且不复用的 object ID。
3. 每帧必须包含 AV。
4. 提供位置、速度、heading 和 agent type。
5. 保证最近 50 帧全部表达在同一个固定 `t0` Ego frame。
6. 若原始 BEV 输出是每帧各自的当前 Ego frame，必须结合定位/里程计在上游完成历史统一重表达。
7. 对时间同步、传感器融合、跟踪 ID 生命周期和坐标质量负责。

### QCNet 预测组接口负责

1. 校验输入类型、shape、数值、频率和坐标标签。
2. 按 object ID 缓存最近 50 帧。
3. 自动构造 QCNet 所需 tensor、`valid_mask`、`agent_type` 和 `av_index`。
4. 执行模型推理，不创建 Trainer、optimizer 或 scheduler。
5. 仅返回内部 60 步输出的前 20 步。
6. 返回输出对象与输入目标 ID 的对应关系。

接口不会自动完成传感器标定、时间同步、跟踪、重采样或 global/逐帧 Ego 到固定 t0 Ego 的坐标变换。

## 3. 最关键的坐标契约

预测时刻 `t0` 是当前 50 帧历史窗口的最后一帧。模型要求整个窗口处于同一个固定
`t0` Ego frame：

- AV 在 `t0` 的位置约为 `[0, 0]` 米。
- AV 在 `t0` 的 heading 约为 `0` 弧度。
- `x` 正方向为 AV 在 `t0` 的前进方向。
- `y` 正方向为 AV 在 `t0` 的左侧方向。
- heading 逆时针为正，单位为弧度。
- position 单位为米。
- velocity 单位为米/秒，并投影在同一固定 Ego 坐标轴上。

禁止把 50 个“每帧分别以当时 AV 为原点”的 frame 直接拼接成历史。这会让坐标系逐帧变化，
与模型训练数据定义不一致。

若 BEV 原始输出为逐帧 Ego 坐标，上游需保存每帧 AV pose，并在调用 QCNet 前将历史状态统一
重表达至当前 `t0`：

```text
历史目标/AV状态（逐帧 Ego 或 global）
          ↓ 定位/里程计与时间对齐
统一转换到当前预测时刻 t0 Ego frame
          ↓
BEVFrame(coordinate_frame="ego")
```

`coordinate_frame="ego"` 是接口契约标签，不会触发坐标变换。接口明确拒绝 `"global"`。

## 4. Python 对象定义

```python
from qcnet_ego import BEVFrame, EgoState, ObjectState

frame = BEVFrame(
    frame_id=10001,
    timestamp=1720000000.0,
    coordinate_frame="ego",
    ego=EgoState(
        id="AV",
        type="vehicle",
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
        heading=0.0,
    ),
    objects=(
        ObjectState(
            id="track_42",
            type="vehicle",
            position=(15.2, 1.3),
            velocity=(3.1, 0.1),
            heading=0.04,
        ),
    ),
)
```

### BEVFrame

| 字段 | 类型 | 必填 | 约束 |
|---|---|---:|---|
| `frame_id` | 可哈希对象 | 是 | 用于上游日志追踪 |
| `timestamp` | float | 是 | 秒；严格递增；相邻约 0.1 秒 |
| `coordinate_frame` | str | 是 | 必须为 `"ego"` |
| `ego` | `EgoState` | 是 | 每帧必须包含 |
| `objects` | `Sequence[ObjectState]` | 是 | 可为空，不包含 AV |

### EgoState / ObjectState

| 字段 | 类型 | 单位 | 约束 |
|---|---|---|---|
| `id` | 可哈希对象 | - | 流内稳定；不得与其他目标重复 |
| `type` | str 或 int | - | 类型名或 0–9 类型索引 |
| `position` | 长度 2 数值序列 | m | `[x,y]`，有限值 |
| `velocity` | 长度 2 数值序列 | m/s | `[vx,vy]`，有限值 |
| `heading` | float | rad | 有限值，接口 wrap 到 `[-pi,pi]` |

支持的类型：

```text
vehicle=0
pedestrian=1
motorcyclist=2
cyclist=3
bus=4
static=5
background=6
construction=7
riderless_bicycle=8
unknown=9
```

未知类别应由 BEV 组显式映射为 `"unknown"`，不要传入任意新字符串。

## 5. 两组组合时的方法调用

### 5.1 服务启动：预测组调用一次

```python
from qcnet_ego import EgoTrajectoryPredictor

predictor = EgoTrajectoryPredictor(
    checkpoint_path="checkpoints/QCNet_Ego_2s_best.ckpt",
    device="cuda",
    return_cpu=True,
)
```

该步骤会：

1. 检查 checkpoint。
2. 严格加载模型权重。
3. 将模型放到指定设备。
4. 执行 `model.eval()`。
5. 创建空的 50 帧 `HistoryBuffer`。

不要每帧重新创建 predictor，否则会反复加载模型并永久无法积满历史窗口。

### 5.2 实时循环：BEV 组每 0.1 秒提交一次

```python
result = predictor.update(bev_frame)
```

方法返回规则：

- 第 1–49 帧：返回 `None`，表示历史窗口尚未就绪。
- 第 50 帧及以后：缓存更新后立即执行一次 QCNet 推理并返回 `PredictionResult`。
- 每次调用滑动保留最新 50 帧。

典型组合代码：

```python
def on_bev_frame(bev_frame: BEVFrame) -> None:
    try:
        result = predictor.update(bev_frame)
    except (ValueError, RuntimeError) as error:
        report_input_error(bev_frame.frame_id, error)
        return

    if result is None:
        report_warming_up(
            current=predictor.history_buffer.size,
            required=50,
        )
        return

    publish_prediction(
        trajectories=result.trajectories,       # [P,6,20,2]
        probabilities=result.probabilities,     # [P,6]
        agent_ids=result.agent_ids,              # 长度 P
        coordinate_frame=result.coordinate_frame,
        reference_time=result.reference_time,
    )
```

### 5.3 输出消费：使用 agent_ids 对齐

不能假设输出第 `i` 个目标永远对应 BEV `objects[i]`。必须使用：

```python
for output_index, agent_id in enumerate(result.agent_ids):
    trajectories = result.trajectories[output_index]   # [6,20,2]
    probabilities = result.probabilities[output_index] # [6]
```

当前帧有效且至少有两个历史采样的目标才进入输出。AV 也会作为一个预测对象返回。

如只消费最高概率轨迹：

```python
import torch

best_mode = result.probabilities.argmax(dim=-1)
best_trajectory = result.trajectories[
    torch.arange(result.trajectories.size(0)), best_mode
]  # [P,20,2]
```

注意：最高概率 mode 是模型的相对选择，不是安全保证；保留全 6 模态更适合下游规划。

### 5.4 场景切换/定位重置：调用 reset_history

以下情况发生时，不应继续沿用原缓存：

- 新 bag、新 sequence 或新车辆会话。
- AV ID 改变。
- 定位系统重置或坐标原点跳变。
- 时间戳倒退。
- 长时间断流，无法可靠重采样到 10 Hz。

调用：

```python
predictor.reset_history()
```

之后重新积累 50 帧。不要通过复用旧 frame 或伪造 valid 数据规避预热。

### 5.5 服务退出：调用 close

```python
predictor.close()
```

该方法清空历史、释放模型引用；CUDA 场景同时请求清理缓存。`close()` 后不能继续 `update()`
或 `predict()`，需要重新构造 predictor。

## 6. 调用时序

```text
预测服务启动
    │
    ├─ EgoTrajectoryPredictor(checkpoint, device) ── 模型严格加载一次
    │
BEV 组 10 Hz 回调
    │
    ├─ 构造 BEVFrame
    ├─ predictor.update(frame)
    │       ├─ 输入校验
    │       ├─ HistoryBuffer.update
    │       ├─ 历史不足 50 帧 → None
    │       └─ 历史达到 50 帧
    │              ├─ to_qcnet_input
    │              ├─ model.eval + torch.inference_mode
    │              └─ PredictionResult
    │
规划/可视化组
    ├─ 按 result.agent_ids 对齐目标
    ├─ 消费 6 模态 × 20 点与概率
    └─ 坐标仍为当前预测 t0 固定 Ego frame
```

## 7. 目标生命周期处理

### 新目标

- 首个有效采样：进入缓存，但暂不进入模型输出。
- 第二个有效采样：自动进入模型和输出。
- 更早历史位置使用零占位，`valid_mask=False`。

### 短暂丢失后恢复

- 丢失帧在历史中为 invalid。
- 如果目标在当前帧恢复，且窗口内累计至少两个有效采样，可重新进入输出。
- ID 必须保持不变。

### 当前帧消失

- 不进入本次 agent 列表和输出。
- 若之后以相同 ID 恢复，保留在仍存在的 50 帧窗口中的历史可继续使用。

### ID switch / ID reuse

- 接口无法从运动值可靠识别 ID switch。
- BEV/跟踪组必须避免把旧 ID 分配给新物体。
- 确认 ID 语义已失效时，应重置相应上游轨迹；当前接口没有单目标删除 API。

## 8. 时间戳和线程约定

- 期望间隔：0.1 秒。
- 默认允许误差：±0.03 秒。
- timestamp 必须严格递增。
- 超出误差时接口抛出 `ValueError`，不自动插值或重采样。
- 当前 predictor/HistoryBuffer 为有状态对象，不保证并发线程安全。
- 同一 predictor 应由单一有序消费循环调用。
- 多路独立场景应分别创建 predictor；不要把不同 scenario 的 frame 混入同一个缓存。

若 BEV 回调与预测线程分离，建议用有界队列串行消费，并制定明确的积压策略。不能乱序调用
`update()`；高负载时也不应静默把不等间隔 frame 当作 10 Hz 输入。

## 9. 输出定义

`PredictionResult`：

| 字段 | Shape/类型 | 说明 |
|---|---|---|
| `trajectories` | `[P,6,20,2]` float tensor | P 个目标的 6 条候选轨迹，单位米 |
| `probabilities` | `[P,6]` float tensor | 每个目标 6 个 mode 的归一化概率 |
| `agent_ids` | `list[str]` | 与 P 维一一对应 |
| `agent_indices` | `[P]` long tensor | 在自动生成 QCNet 输入中的索引 |
| `coordinate_frame` | `"ego"` | 固定 t0 Ego frame |
| `reference_time` | `"history_last_t0"` | 参考时刻 |
| `horizon_steps` | `20` | 对外时间步数 |
| `horizon_seconds` | `2.0` | 预测时域 |
| `frequency_hz` | `10.0` | 输出频率 |

时间点依次为 `t0+0.1s` 至 `t0+2.0s`。默认 `return_cpu=True`，输出 tensor 会回到 CPU；
若同机 CUDA 下游希望避免拷贝，可在初始化时设置 `return_cpu=False`，但下游必须正确管理
CUDA tensor 生命周期和同步。

## 10. 异常与联调日志

建议 BEV 组记录：

- frame_id、timestamp。
- 当前 AV ID 和目标数量。
- 坐标转换版本/定位状态。
- object ID、type 及有效性。

建议预测组记录：

- 当前缓存帧数。
- 本次输入/输出 agent IDs。
- 推理耗时和设备。
- 输入异常消息。
- 是否发生 reset。

常见异常：

| 异常 | 原因 | 处理 |
|---|---|---|
| coordinate frame 被拒绝 | 不是 `"ego"` | 上游完成固定 t0 Ego 变换 |
| timestamp 被拒绝 | 倒退或非 10 Hz | 检查同步/重采样，必要时 reset |
| AV pose 被拒绝 | t0 不在原点或 heading 非 0 | 检查坐标变换，禁止绕过校验 |
| duplicate object ID | 同帧 ID 重复或与 AV 重复 | 修复跟踪输出 |
| agent type changed | 同 ID 在窗口内类别变化 | 固定类别策略或重置流 |
| CUDA unavailable | 环境或 device 配置错误 | 检查 PyTorch/CUDA，或改用 CPU |

## 11. 最小联合验证流程

1. 先运行随包人工 demo：

```bat
python -B examples\demo_bev_input.py --device cpu
python -B examples\demo_bev_input.py --device cuda
```

2. 用真实 BEV 录制的小段数据离线回放，确认：

- 50 帧均为固定 t0 Ego frame。
- 10 Hz 时间戳单调。
- AV t0 pose 合法。
- ID 连续。
- 第 50 帧开始得到 `[P,6,20,2]` 和 `[P,6]`。
- 概率每行和约为 1。
- `agent_ids` 与可视化目标一致。

3. 再接实时进程，监控输入队列、推理耗时、丢帧和 reset。

4. 集成验收至少覆盖：直行、转弯、静止、目标新生/消失、短暂遮挡、定位重置、时间戳异常。

## 12. 已验证状态

当前交付版已在以下环境完成 CPU 和 CUDA 的 50 帧实时接口 smoke test：

- Python 3.10.20
- PyTorch 2.0.1
- CUDA 11.8
- torch-geometric 2.3.0

验证输出：

```text
Input agents: 4
Output trajectories: [4, 6, 20, 2]
Output probabilities: [4, 6]
Probability sums: approximately 1
Coordinate frame: ego
Horizon: 20 steps / 2.0 seconds
```

该验证证明接口、缓存、模型加载和 tensor shape 可运行，不代表真实道路安全性能验证。
