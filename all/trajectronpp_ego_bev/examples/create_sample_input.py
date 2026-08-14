"""Create synthetic input for API validation only, not prediction-quality testing."""

from pathlib import Path

import torch


def main() -> None:
    num_agents = 4
    time_seconds = (torch.arange(50, dtype=torch.float32) - 49.0) / 10.0
    position = torch.zeros(num_agents, 50, 2, dtype=torch.float32)
    velocity = torch.zeros_like(position)
    heading = torch.zeros(num_agents, 50, dtype=torch.float32)

    # All coordinates are already in the fixed AV-at-t0 frame.
    position[0, :, 0] = time_seconds * 1.0
    velocity[0, :, 0] = 1.0
    position[1, :, 0] = 10.0 + time_seconds * 0.5
    position[1, :, 1] = 2.0
    velocity[1, :, 0] = 0.5
    position[2, :, 0] = -5.0 + time_seconds * 1.2
    position[2, :, 1] = -3.0
    velocity[2, :, 0] = 1.2
    position[3, :, 0] = 4.0
    position[3, :, 1] = 5.0 + time_seconds * 0.1
    velocity[3, :, 1] = 0.1
    heading[3] = torch.pi / 2

    sample = {
        'position': position,
        'velocity': velocity,
        'heading': heading,
        'valid_mask': torch.ones(num_agents, 50, dtype=torch.bool),
        'agent_type': torch.tensor([0, 0, 0, 1], dtype=torch.long),
        'av_index': 0,
        'agent_ids': ['AV', 'vehicle_1', 'vehicle_2', 'pedestrian_1'],
        'coordinate_frame': 'ego',
        'frequency_hz': 10.0,
    }
    output_path = Path(__file__).resolve().parent / 'sample_input.pt'
    torch.save(sample, output_path)
    print('Synthetic interface-validation sample created.')
    print(f'Path: {output_path}')
    print(f'Input agents: {num_agents}')
    print(f'History shape: {list(position.shape)}')


if __name__ == '__main__':
    main()
