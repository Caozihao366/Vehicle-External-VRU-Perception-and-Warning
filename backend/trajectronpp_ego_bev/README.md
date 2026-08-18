# QCNet Ego 2 秒轨迹预测部署包

## 1. 项目说明

这是一个仅用于推理集成的 **Ego-coordinate、map-free QCNet** 轨迹预测包。它不是
Trajectron++；`trajectronpp_ego_bev` 只是沿用上层工程已有目录名。

本目录不包含训练入口、数据集、AV2 SDK、Lightning Trainer、loss、metrics、optimizer、
scheduler 或训练日志。本交付是“可集成推理初版”，不代表已经完成真实车辆安全部署。

## 2. 模型能力

随包 checkpoint 是仅用于推理的 weights-only 副本：保留模型权重和必要模型配置，不包含 optimizer、scheduler、epoch、global step、GradScaler 或 callback 状态。

- 输入：多目标最近 50 步历史，10 Hz，共 5 秒。
- 输入坐标：预测起点 `t0` 的固定 AV Ego frame。
- 地图：不使用 HD Map。
- Ego 特征：启用 `ego_relative_emb` 和 `ego_gate`。
- 内部 Decoder：60 步，保留以严格兼容 checkpoint。
- 对外输出：前 20 步，10 Hz，未来 2 秒。
- 模态：每个被预测目标返回 6 条候选轨迹及 6 个归一化概率。

## 3. 坐标系定义

参考时刻 `t0` 是历史第 50 帧，即历史最后一帧。输入和输出始终位于同一个固定
`t0` Ego frame：

- AV 在 `t0` 的位置必须约为 `[0, 0]` 米。
- AV 在 `t0` 的 heading 必须约为 `0` 弧度。
- `x` 正方向是 AV 在 `t0` 的前进方向。
- `y` 正方向是 AV 在 `t0` 的左侧方向。
- heading 为相对 `t0` AV 航向的弧度，逆时针为正，并按 `[-pi, pi]` 解释。
- position 单位为米，velocity 单位为米/秒。

上述轴向来自训练代码的变换：`x = cos(theta)*dx + sin(theta)*dy`，
`y = -sin(theta)*dx + cos(theta)*dy`。所有历史点和未来点都使用固定 `t0` frame；
**不是**在每个时刻重新把 AV 移到原点。

本交付版只接收已经转换好的 Ego 输入，不提供 global 自动转换。代码能检查 AV 的 t0
pose 和坐标标签，但无法仅凭数值证明调用者没有把 global 数据伪装成 Ego 数据；该契约
必须由上游保证。

## 4. 环境安装

测试环境：

- Windows 64-bit
- Python 3.10.20
- PyTorch 2.0.1
- CUDA 11.8
- torch-geometric 2.3.0
- torch-cluster 1.6.1+pt20cu118
- torch-scatter 2.1.1+pt20cu118
- torch-sparse 0.6.17+pt20cu118

推理代码不依赖 PyTorch Lightning、AV2 SDK、pandas 或 TensorBoard。

### CUDA 11.8

```bat
conda env create -f environment.yml
conda activate qcnet-ego-inference
pip install torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

### CPU

```bat
conda create -n qcnet-ego-cpu python=3.10 -y
conda activate qcnet-ego-cpu
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.1 -f https://data.pyg.org/whl/torch-2.0.1+cpu.html
pip install torch-geometric==2.3.0
```

PyG 扩展轮必须与 PyTorch/CUDA 版本匹配。Windows 上不要混装不同 CUDA 后缀的 wheel。

## 5. 快速开始

先生成不代表真实预测质量的人工接口样本：

```bat
python examples\create_sample_input.py
```

CPU demo：

```bat
python examples\demo_predict.py --device cpu
```

CUDA demo：

```bat
python examples\demo_predict.py --device cuda
```

实时 BEV 对象接口 demo：

```bat
python examples\demo_bev_input.py --device cpu
python examples\demo_bev_input.py --device cuda
```

## 6. Python 接口

```python
from pathlib import Path
import torch
from qcnet_ego import EgoTrajectoryPredictor

root = Path.cwd()
sample = torch.load(root / 'examples' / 'sample_input.pt', map_location='cpu')
predictor = EgoTrajectoryPredictor(
    checkpoint_path=str(root / 'checkpoints' / 'QCNet_Ego_2s_best.ckpt'),
    device='cuda',
    return_cpu=True,
)
result = predictor.predict(sample)
print(result.trajectories.shape)   # [P, 6, 20, 2]
print(result.probabilities.shape)  # [P, 6]
predictor.close()
```

接口一次只接受一个 scenario。没有提供 `predict_batch()`，以避免 PyG batch 中 local/global
`av_index` 语义歧义。

### 实时 BEV 接口

```python
from qcnet_ego import BEVFrame, EgoState, ObjectState, RealtimeTrajectoryPredictor

predictor = RealtimeTrajectoryPredictor(
    'checkpoints/QCNet_Ego_2s_best.ckpt', device='cuda'
)

for frame in realtime_bev_frames:
    result = predictor.update(frame)
    if result is not None:  # 至少两个有效采样后，每个新帧产生一次预测
        consume(result.trajectories, result.probabilities)
```

`BEVFrame` 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `frame_id` | 任意可哈希 ID | 上游帧标识 |
| `timestamp` | float | 单调递增秒时间戳，帧间隔应为 0.1 秒 |
| `coordinate_frame` | str | 必须为 `"ego"` |
| `ego` | `EgoState` | 必须包含且 ID 在同一流中稳定 |
| `objects` | `Sequence[ObjectState]` | 当前帧可见目标 |

`EgoState` 包含 `id`、`position=[x,y]`、`velocity=[vx,vy]`、`heading`；自车类型固定映射为
vehicle。`ObjectState` 另含 `type`，支持规定类型名；无法识别的类别统一映射为 unknown。

`BEVHistoryBuffer` 按 object ID 保存最近 50 帧。历史中缺失的采样由零值占位并令
`valid_mask=False`；不足 50 帧时在左侧补 invalid。第 1 帧 `update()` 返回 `None`，AV 达到
两个有效采样后即可预测，不要求等满 50 帧。新序列开始前调用 `predictor.reset_history()`。
刚出现且只有一个有效采样的目标会被暂缓，不影响其他目标预测；获得第二个有效采样后自动
进入模型和输出。输出的 `agent_ids`/`agent_indices` 是本次实际预测对象的对应关系。

**坐标契约：接口不会执行 SE(2) 变换。缓存窗口内所有 position、velocity、heading 必须由
BEV/定位上游预先表达在同一个固定预测参考 Ego frame。不能直接把“每帧各自以当时 AV 为
原点”的 50 组数值拼接起来；那会形成随时间变化的坐标系，与模型训练输入不一致。若 BEV
只输出逐帧当前 Ego frame，上游必须先借助 ego pose/里程计把历史窗口统一重表达，再调用
本接口。坐标转换可通过 `CoordinateWindowAdapter` 插件接入；默认 identity adapter 不做变换。
`demo_bev_input.py` 使用静止 AV，因此其 50 帧天然共享同一参考 frame，仅用于接口
与 tensor shape 验证，不代表真实预测质量。**

## 7. 输入字段、shape 与 dtype

| 字段 | Shape/类型 | dtype | 单位 | 必填 | 说明 |
|---|---|---|---|---|---|
| `position` | `[N,50,2]` | float32 | m | 是 | 固定 t0 Ego frame 中的位置 |
| `velocity` | `[N,50,2]` | float32 | m/s | 是 | 固定 t0 Ego 轴上的速度向量 |
| `heading` | `[N,50]` | float32 | rad | 是 | 相对 t0 AV 航向 |
| `valid_mask` | `[N,50]` | bool | - | 是 | 对应历史采样是否有效 |
| `agent_type` | `[N]` | int64 | - | 是 | AV2 类型索引 0–9 |
| `av_index` | Python int | - | - | 是 | AV 在输入 agent 列表中的索引 |
| `agent_ids` | `list[str]`，长度 N | - | - | 否 | 上游稳定目标 ID |
| `coordinate_frame` | `"ego"` | - | - | 是 | 其他值直接拒绝 |
| `frequency_hz` | `10.0` | float | Hz | 是 | 不支持自动重采样 |

`agent_type` 顺序：`vehicle=0, pedestrian=1, motorcyclist=2, cyclist=3, bus=4,
static=5, background=6, construction=7, riderless_bicycle=8, unknown=9`。

适配器会自动生成 PyG `HeteroData`、`num_nodes` 和 decoder 所需 `predict_mask`。调用者不需要
提供 `ptr`、`batch`、edge index、category 或地图字段。当前 t0 有效的 agent 被标为预测对象；
t0 无效的 agent 仍可作为历史上下文，但不会出现在最终轨迹数组中。

校验包括 shape、N 一致性、类型范围、NaN/Inf、AV 索引、10 Hz、坐标标签、AV t0 pose、
当前有效性以及被预测 agent 至少两个有效历史采样。历史不足 50 帧时必须由上游补齐到
50 帧，并用 `valid_mask=False` 标记无效位置；不得伪造历史。

## 8. 输出 shape 与含义

`PredictionResult` 包含：

| 字段 | Shape/类型 | 单位 | 说明 |
|---|---|---|---|
| `trajectories` | `[P,6,20,2]` | m | P 个 t0 有效目标的 6 模态轨迹 |
| `probabilities` | `[P,6]` | - | softmax 概率，每行和约等于 1 |
| `agent_indices` | `[P]` | - | 输出目标在原输入 N 个 agent 中的索引 |
| `agent_ids` | `list[str]` 或 None | - | 与输出 P 维一一对应 |
| `coordinate_frame` | `"ego"` | - | 输出仍是固定 t0 Ego frame |
| `reference_time` | `"history_last_t0"` | - | 坐标参考时刻 |
| `horizon_steps` | `20` | step | 对外未来步数 |
| `horizon_seconds` | `2.0` | s | 对外未来时域 |
| `frequency_hz` | `10.0` | Hz | 输出频率 |

时间维依次对应 `t0+0.1s, t0+0.2s, ..., t0+2.0s`。mode 维不是时间维；它表示 6 个
互斥候选。概率是模型相对置信度，不是安全概率或碰撞概率。默认 `return_cpu=True`，因此
结果 tensor 自动回到 CPU。可用 `result.to_dict()` 或 `result.save(path)`。

最高概率轨迹示例：

```python
best_mode = result.probabilities.argmax(dim=-1)
best_trajectory = result.trajectories[
    torch.arange(result.trajectories.size(0)), best_mode
]
result.save('my_prediction.pt')
```

## 9. 文件输入示例

```python
import torch
from qcnet_ego import EgoTrajectoryPredictor

input_data = torch.load('examples/sample_input.pt', map_location='cpu')
predictor = EgoTrajectoryPredictor(
    'checkpoints/QCNet_Ego_2s_best.ckpt', device='cpu'
)
result = predictor.predict(input_data)
print(result.trajectories.shape)
print(result.probabilities.sum(-1))
result.save('examples/sample_output.pt')
```

## 10. 上游 BEV/多目标跟踪集成

- 为每个目标维护按 10 Hz 对齐的 50 帧缓存。
- 同一物理目标必须保持稳定 ID；ID 跳变会切断运动历史。
- AV 必须出现在 agent 列表中，`av_index` 指向它。
- 在统一 t0 使用 AV pose 对所有 agent 的历史位置、速度和 heading 做一次 SE(2) 转换。
- 不要逐时间点重新建立 Ego frame。
- 新出现目标：缺失历史位置可填零，但对应 `valid_mask=False`；至少两个有效点后才预测。
- 暂时丢失目标：保留历史和 false mask；若 t0 无效，仅作为上下文，不输出预测。
- 彻底消失目标：从缓存淘汰前应确保不会导致 agent ID 被错误复用。
- 上游若不是 10 Hz，必须在进入本接口前完成受控重采样和有效性标注。

## 11. 错误处理

- shape 或 dtype 无法转换：`ValueError`，消息包含字段名和实际 shape。
- `coordinate_frame != "ego"`：拒绝，不自动猜测或转换。
- AV t0 不在原点/heading 不为 0：拒绝。
- checkpoint 缺失：`FileNotFoundError`。
- CUDA 不可用但请求 CUDA：`ValueError`。
- 输入或输出包含 NaN/Inf：拒绝。
- checkpoint 配置不符、ego 权重缺失、Decoder shape mismatch：严格加载失败。
- 历史 tensor 不是 50 步：拒绝。

## 12. 性能说明

当前记录来自固定 5000 场景的 AV2 validation 子集，不是 AV2 完整官方榜单：

- minADE: 0.7801 m
- minFDE: 0.8872 m
- MR: 0.1218

这些结果不能用于承诺真实道路安全性能，也不能直接与不同预测时域或完整榜单结果比较。

## 13. 已知限制

- 无 HD Map 输入。
- 只返回未来 2 秒。
- 训练分布来自 Argoverse 2。
- 输入必须严格符合固定 t0 Ego frame。
- 不同传感器、频率、类别分布或车辆动力学可能需要重新适配。
- minADE/minFDE 是 6 模态中的 min 指标，不代表最高概率轨迹一定达到该误差。
- mode 概率校准仍有改进空间。
- 当前只支持单 scenario 调用。
- 输入合法性校验不能替代上游完整坐标变换单元测试。

## 14. 目录说明

```text
checkpoints/       最佳模型 checkpoint
configs/           推理元数据配置
qcnet_ego/         纯 PyTorch 模型、输入适配和高级接口
examples/          人工输入生成、demo 与样例输出
model_info.json    模型卡片式机器可读信息
requirements.txt  最小 Python 依赖
environment.yml   Conda 环境参考
LICENSE            原 QCNet Apache-2.0 许可证
LICENSE_NOTES.md   来源、引用和第三方许可证说明
```

## 15. 模型来源、引用和许可证

模型基于 Zikang Zhou 等人的 QCNet（CVPR 2023）代码和论文：

```bibtex
@inproceedings{zhou2023query,
  title={Query-Centric Trajectory Prediction},
  author={Zhou, Zikang and Wang, Jianping and Li, Yung-Hui and Huang, Yu-Kai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

源 QCNet 代码使用 Apache License 2.0，详见 `LICENSE` 和 `LICENSE_NOTES.md`。训练数据来源
于 Argoverse 2；本交付不包含 AV2 数据或 AV2 SDK。使用方仍需自行遵守模型、依赖、数据和
部署地区适用的许可证及法规。
