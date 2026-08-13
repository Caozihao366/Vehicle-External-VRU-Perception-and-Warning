"""Feed 50 synthetic BEV frames into the real-time predictor interface."""

from argparse import ArgumentParser
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from qcnet_ego import BEVFrame, EgoState, EgoTrajectoryPredictor, ObjectState


def create_frame(frame_index: int) -> BEVFrame:
    """Create interface-only data in one fixed Ego frame with a stationary AV."""
    time_seconds = frame_index / 10.0
    return BEVFrame(
        frame_id=frame_index,
        timestamp=time_seconds,
        coordinate_frame='ego',
        ego=EgoState(
            id='AV', type='vehicle', position=(0.0, 0.0),
            velocity=(0.0, 0.0), heading=0.0,
        ),
        objects=(
            ObjectState(
                id='vehicle_front', type='vehicle',
                position=(12.0 + 1.5 * time_seconds, 1.5),
                velocity=(1.5, 0.0), heading=0.0,
            ),
            ObjectState(
                id='vehicle_left', type='vehicle',
                position=(5.0 + 0.8 * time_seconds, 4.0),
                velocity=(0.8, 0.0), heading=0.0,
            ),
            ObjectState(
                id='pedestrian_right', type='pedestrian',
                position=(8.0, -5.0 + 0.5 * time_seconds),
                velocity=(0.0, 0.5), heading=1.5707963267948966,
            ),
        ),
    )


def main() -> None:
    parser = ArgumentParser(description='Run QCNet with 50 real-time-style BEV frames.')
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument(
        '--checkpoint',
        default=str(PACKAGE_ROOT / 'checkpoints' / 'QCNet_Ego_2s_best.ckpt'),
    )
    args = parser.parse_args()

    predictor = EgoTrajectoryPredictor(args.checkpoint, device=args.device)
    result = None
    for frame_index in range(50):
        result = predictor.update(create_frame(frame_index))
        if frame_index < 49:
            assert result is None

    if result is None:
        raise RuntimeError('No prediction was produced after 50 frames.')
    print('Buffered frames:', predictor.history_buffer.size)
    print('Input agents: 4')
    print('Output trajectories:', list(result.trajectories.shape))
    print('Output probabilities:', list(result.probabilities.shape))
    print('Probability sums:', result.probabilities.sum(dim=-1).tolist())
    print('Agent IDs:', result.agent_ids)
    print('Coordinate frame:', result.coordinate_frame)
    print(f'Horizon: {result.horizon_steps} steps / {result.horizon_seconds:.1f} seconds')
    predictor.close()


if __name__ == '__main__':
    main()
