"""测试 QCNet 模型加载和推理。"""
import os, sys
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time

# 添加 qcnet 路径
qcnet_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trajectronpp_ego_bev')
sys.path.insert(0, qcnet_parent)

print('Loading QCNet model...')
t0 = time.time()
from qcnet_ego import EgoTrajectoryPredictor
predictor = EgoTrajectoryPredictor(
    checkpoint_path=os.path.join(qcnet_parent, 'checkpoints', 'QCNet_Ego_2s_best.ckpt'),
    device='cuda',
)
print(f'Model loaded in {time.time()-t0:.1f}s')

# 构造测试输入：2个agent（ego + 1个行人），50步历史
N = 2
H = 50
position = np.zeros((N, H, 2), dtype=np.float32)
velocity = np.zeros((N, H, 2), dtype=np.float32)
heading = np.zeros((N, H), dtype=np.float32)
valid_mask = np.zeros((N, H), dtype=bool)

# ego: 静止在原点
position[0, :] = [0, 0]
velocity[0, :] = [0, 0]
heading[0, :] = 0
valid_mask[0, :] = True

# 行人：在前方10m处，以1m/s向ego移动
for t in range(H):
    position[1, t] = [10.0 - t * 0.1, 0.0]
    velocity[1, t] = [-1.0, 0.0]
    heading[1, t] = np.pi
    valid_mask[1, t] = True

input_data = {
    'position': position,
    'velocity': velocity,
    'heading': heading,
    'valid_mask': valid_mask,
    'agent_type': np.array([0, 1], dtype=np.int64),  # 0=vehicle, 1=pedestrian
    'av_index': 0,
    'coordinate_frame': 'ego',
    'frequency_hz': 10.0,
}

print('Running inference...')
t0 = time.time()
result = predictor.predict(input_data)
print(f'Inference in {time.time()-t0:.3f}s')
print(f'trajectories shape: {result.trajectories.shape}')
print(f'probabilities shape: {result.probabilities.shape}')
print(f'probabilities: {result.probabilities}')
print(f'horizon: {result.horizon_seconds}s, modes: {result.num_modes}')
print(f'agent_indices: {result.agent_indices}')

# 打印预测轨迹（第一个agent的最可能模式）
traj = result.trajectories[0, 0, :, :].cpu().numpy()  # agent 0, mode 0
print(f'Agent 0 mode 0 first 5 points: {traj[:5]}')
print('QCNet test passed!')
