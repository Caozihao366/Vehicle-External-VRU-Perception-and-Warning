from argparse import ArgumentParser
from pathlib import Path
import sys

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from qcnet_ego import EgoTrajectoryPredictor


def main() -> None:
    parser = ArgumentParser(description='Run one QCNet Ego 2-second prediction.')
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument(
        '--checkpoint',
        default=str(PACKAGE_ROOT / 'checkpoints' / 'QCNet_Ego_2s_best.ckpt'),
    )
    parser.add_argument(
        '--input',
        default=str(PACKAGE_ROOT / 'examples' / 'sample_input.pt'),
    )
    parser.add_argument(
        '--output',
        default=str(PACKAGE_ROOT / 'examples' / 'sample_output.pt'),
    )
    args = parser.parse_args()

    sample = torch.load(Path(args.input).expanduser().resolve(), map_location='cpu')
    predictor = EgoTrajectoryPredictor(args.checkpoint, device=args.device, return_cpu=True)
    result = predictor.predict(sample)
    result.save(args.output)

    print(
        f'Checkpoint strict load: {predictor.load_audit["strict"]}; '
        f'state tensors: {predictor.load_audit["state_tensor_count"]}; '
        f'ego tensors: {predictor.load_audit["ego_tensor_count"]}')
    print(f'Input agents: {sample["position"].size(0)}')
    print(f'History shape: {list(sample["position"].shape)}')
    print(f'Output trajectories: {list(result.trajectories.shape)}')
    print(f'Output probabilities: {list(result.probabilities.shape)}')
    print(f'Probability sums: {result.probabilities.sum(dim=-1).tolist()}')
    print(f'Coordinate frame: {result.coordinate_frame}')
    print(f'Horizon: {result.horizon_steps} steps / {result.horizon_seconds:.1f} seconds')
    print(f'Predicted input indices: {result.agent_indices.tolist()}')
    print(f'Output saved: {Path(args.output).expanduser().resolve()}')
    predictor.close()


if __name__ == '__main__':
    main()
