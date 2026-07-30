# Trajectron++ Ego-BEV 推理包（旧权重 / mini）

用于并入主流程的最小推理包：自车 BEV 坐标、无地图、`int_ee`。

## 包内容

```
trajectronpp_ego_bev/
  weights/
    config.json
    model_registrar-200.pt      # mini 训出的旧权重
  code/trajectron/              # 推理所需代码
  inference/
    ego_bev_predictor.py        # 主流程调用入口
  examples/smoke_predict.py
  env_meta.json
  requirements_infer.txt
  README.md
```

## 模型信息

| 项 | 值 |
|---|---|
| 权重 | `int_ee_ego_bev` epoch **200** |
| 训练数据 | nuScenes **mini** ego-BEV |
| 坐标 | 当前自车 BEV：x 前、y 左（米） |
| 时间 | `dt=0.5s` |
| 默认预测 | **1.5s**（3 步：0.5 / 1.0 / 1.5） |
| 地图 | 不需要 |
| mini@1.5s | 车辆 ADE≈1.51m / FDE≈2.59m |

## 主流程最小接入

```python
import sys
sys.path.insert(0, 'path/to/trajectronpp_ego_bev/inference')
from ego_bev_predictor import EgoBevPredictor

predictor = EgoBevPredictor(
    package_root='path/to/trajectronpp_ego_bev',
    checkpoint=200,
    device='cpu',   # 有 GPU 可改 'cuda:0'
)

# 所有 history 必须已在【同一当前自车 BEV】下
agents = [
    {
        'id': 'ego',
        'type': 'VEHICLE',
        'history_xy': ego_xy,             # (T, 2) float, 米；建议 T>=2，最多用 8
        'history_heading': ego_heading,   # (T,) rad；可选但建议有
        'is_ego': True,
    },
    {
        'id': 'veh_1',
        'type': 'VEHICLE',                # 或 'PEDESTRIAN'
        'history_xy': veh_xy,
        'history_heading': veh_heading,
    },
]

preds = predictor.predict(agents, prediction_horizon_sec=1.5)
# preds['veh_1'].shape == (1, 3, 2)
# 绝对自车 BEV 坐标，对应未来 0.5s / 1.0s / 1.5s
```

## 输入约定（必须一致）

1. 原点：当前自车  
2. x 前、y 左，单位米  
3. heading：弧度，0=+x，逆时针为正  
4. 多帧历史都变换到**当前**自车系（不是各自当时自车系）  
5. 建议 2Hz（与训练 `dt=0.5` 一致）

## 依赖

见 `requirements_infer.txt`（Python 3.6 + torch 1.4 验证过）。

冒烟：

```bash
cd trajectronpp_ego_bev
python examples/smoke_predict.py
```
