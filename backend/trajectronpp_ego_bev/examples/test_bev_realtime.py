"""End-to-end smoke test for the real-time BEV deployment adapter."""

from argparse import ArgumentParser
from pathlib import Path
import sys

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from qcnet_ego import BEVFrame, EgoState, ObjectState, RealtimeTrajectoryPredictor


def create_frame(frame_index: int) -> BEVFrame:
    time_seconds = frame_index / 10.0
    objects = [
        ObjectState(
            id='vehicle_001', type='vehicle',
            position=[10.0 + 1.2 * time_seconds, 1.0],
            velocity=[1.2, 0.0], heading=0.0,
        ),
        ObjectState(
            id='pedestrian_001', type='pedestrian',
            position=[7.0, -3.0 + 0.4 * time_seconds],
            velocity=[0.0, 0.4], heading=1.5707963267948966,
        ),
    ]
    # Human frame numbers 20-24 correspond to zero-based indices 19-23.
    if not 19 <= frame_index <= 23:
        objects.insert(1, ObjectState(
            id='vehicle_002', type='vehicle',
            position=[5.0 + 0.7 * time_seconds, 4.0],
            velocity=[0.7, 0.0], heading=0.0,
        ))
    return BEVFrame(
        frame_id=frame_index + 1,
        timestamp=time_seconds,
        coordinate_frame='ego',
        ego=EgoState(
            id='AV', position=[0.0, 0.0],
            velocity=[0.0, 0.0], heading=0.0,
        ),
        objects=objects,
    )


def assert_prediction(result) -> None:
    assert result.trajectories.ndim == 4
    assert tuple(result.trajectories.shape[-3:]) == (6, 20, 2)
    assert tuple(result.probabilities.shape) == (
        result.trajectories.size(0), 6)
    assert torch.isfinite(result.trajectories).all()
    assert torch.isfinite(result.probabilities).all()
    assert torch.allclose(
        result.probabilities.sum(dim=-1),
        torch.ones(result.probabilities.size(0)),
        atol=1e-5,
        rtol=1e-5,
    )


def main() -> None:
    parser = ArgumentParser(description='Test 60 real-time BEV frames with QCNet.')
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument(
        '--checkpoint',
        default=str(PACKAGE_ROOT / 'checkpoints' / 'QCNet_Ego_2s_best.ckpt'),
    )
    args = parser.parse_args()

    realtime_predictor = RealtimeTrajectoryPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        return_cpu=True,
    )
    result = None
    frame_50_input = None
    missing_output_checked = False
    for frame_index in range(60):
        result = realtime_predictor.update(create_frame(frame_index))
        if frame_index == 0:
            assert realtime_predictor.history_buffer.num_frames == 1
            assert result is None
        else:
            assert result is not None
            assert_prediction(result)
        if 19 <= frame_index <= 23 and result is not None:
            assert 'vehicle_002' not in result.agent_ids
            missing_output_checked = True
        if frame_index == 49:
            frame_50_input = realtime_predictor.history_buffer.build_model_input()

    assert result is not None
    assert frame_50_input is not None
    assert missing_output_checked
    expected_ids = ['AV', 'vehicle_001', 'vehicle_002', 'pedestrian_001']
    assert frame_50_input['agent_ids'] == expected_ids
    assert frame_50_input['av_index'] == 0
    assert tuple(frame_50_input['position'].shape) == (4, 50, 2)
    assert tuple(frame_50_input['velocity'].shape) == (4, 50, 2)
    assert tuple(frame_50_input['heading'].shape) == (4, 50)
    assert tuple(frame_50_input['valid_mask'].shape) == (4, 50)
    vehicle_002_index = expected_ids.index('vehicle_002')
    missing_mask = frame_50_input['valid_mask'][vehicle_002_index, 19:24]
    assert not missing_mask.any()
    assert int(frame_50_input['valid_mask'][vehicle_002_index].sum()) == 45
    assert 'vehicle_002' in realtime_predictor.history_buffer.track_ids
    assert result.agent_ids == expected_ids

    print('Frames processed: 60')
    print('Buffered frames:', realtime_predictor.history_buffer.num_frames)
    print('Track IDs:', realtime_predictor.history_buffer.track_ids)
    print('Frame-50 position:', list(frame_50_input['position'].shape))
    print('Frame-50 velocity:', list(frame_50_input['velocity'].shape))
    print('Frame-50 heading:', list(frame_50_input['heading'].shape))
    print('Frame-50 valid_mask:', list(frame_50_input['valid_mask'].shape))
    print('vehicle_002 missing valid count:', int(missing_mask.sum()))
    print('AV index:', frame_50_input['av_index'])
    print('Predicted IDs:', result.agent_ids)
    print('Trajectories:', list(result.trajectories.shape))
    print('Probabilities:', list(result.probabilities.shape))
    print('Probability sums:', result.probabilities.sum(dim=-1).tolist())
    realtime_predictor.close()


if __name__ == '__main__':
    main()
