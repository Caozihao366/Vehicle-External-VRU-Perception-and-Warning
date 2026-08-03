"""Validate a deployment-friendly dictionary and build one PyG scenario."""

from dataclasses import dataclass
from typing import List, Mapping, Optional

import torch
from torch_geometric.data import HeteroData


AGENT_TYPES = (
    'vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static',
    'background', 'construction', 'riderless_bicycle', 'unknown',
)


@dataclass(frozen=True)
class AdaptedInput:
    data: HeteroData
    prediction_indices: torch.Tensor
    prediction_agent_ids: Optional[List[str]]


class EgoInputAdapter:
    history_steps = 50
    frequency_hz = 10.0
    pose_tolerance = 1e-4

    @staticmethod
    def _tensor(mapping: Mapping[str, object], key: str, dtype: torch.dtype) -> torch.Tensor:
        if key not in mapping:
            raise ValueError(f'Missing required input field: {key}')
        try:
            return torch.as_tensor(mapping[key], dtype=dtype).detach().cpu().contiguous()
        except (TypeError, ValueError) as error:
            raise ValueError(f'Input field {key!r} cannot be converted to a tensor.') from error

    def adapt(self, input_data: Mapping[str, object]) -> AdaptedInput:
        if not isinstance(input_data, Mapping):
            raise ValueError('input_data must be a mapping/dictionary.')

        position = self._tensor(input_data, 'position', torch.float32)
        velocity = self._tensor(input_data, 'velocity', torch.float32)
        heading = self._tensor(input_data, 'heading', torch.float32)
        valid_mask = self._tensor(input_data, 'valid_mask', torch.bool)
        agent_type = self._tensor(input_data, 'agent_type', torch.long)

        if position.ndim != 3 or tuple(position.shape[1:]) != (self.history_steps, 2):
            raise ValueError(f'position must have shape [N, 50, 2], got {tuple(position.shape)}.')
        num_agents = position.size(0)
        expected_vector_shape = (num_agents, self.history_steps, 2)
        expected_scalar_shape = (num_agents, self.history_steps)
        if tuple(velocity.shape) != expected_vector_shape:
            raise ValueError(f'velocity must have shape {expected_vector_shape}, got {tuple(velocity.shape)}.')
        if tuple(heading.shape) != expected_scalar_shape:
            raise ValueError(f'heading must have shape {expected_scalar_shape}, got {tuple(heading.shape)}.')
        if tuple(valid_mask.shape) != expected_scalar_shape:
            raise ValueError(f'valid_mask must have shape {expected_scalar_shape}, got {tuple(valid_mask.shape)}.')
        if tuple(agent_type.shape) != (num_agents,):
            raise ValueError(f'agent_type must have shape [{num_agents}], got {tuple(agent_type.shape)}.')
        if num_agents < 1:
            raise ValueError('At least one agent is required.')
        if not torch.isfinite(position).all():
            raise ValueError('position contains NaN or Inf.')
        if not torch.isfinite(velocity).all():
            raise ValueError('velocity contains NaN or Inf.')
        if not torch.isfinite(heading).all():
            raise ValueError('heading contains NaN or Inf.')
        if ((agent_type < 0) | (agent_type >= len(AGENT_TYPES))).any():
            raise ValueError(f'agent_type values must be in [0, {len(AGENT_TYPES) - 1}].')

        try:
            av_index = int(input_data['av_index'])
        except KeyError as error:
            raise ValueError('Missing required input field: av_index') from error
        except (TypeError, ValueError) as error:
            raise ValueError('av_index must be an integer.') from error
        if not 0 <= av_index < num_agents:
            raise ValueError(f'av_index must be in [0, {num_agents - 1}], got {av_index}.')

        coordinate_frame = input_data.get('coordinate_frame')
        if coordinate_frame != 'ego':
            raise ValueError(
                "coordinate_frame must be 'ego'. This delivery package does not "
                'perform global-to-ego conversion.')
        try:
            frequency_hz = float(input_data.get('frequency_hz'))
        except (TypeError, ValueError) as error:
            raise ValueError('frequency_hz is required and must be numeric.') from error
        if abs(frequency_hz - self.frequency_hz) > 1e-6:
            raise ValueError('frequency_hz must be exactly 10.0; automatic resampling is not supported.')

        if not bool(valid_mask[av_index, -1]):
            raise ValueError('The AV must be valid at the final history timestep t0.')
        if position[av_index, -1].abs().max().item() > self.pose_tolerance:
            raise ValueError('AV position at t0 must be approximately [0, 0] in the fixed Ego frame.')
        wrapped_av_heading = torch.atan2(
            heading[av_index, -1].sin(), heading[av_index, -1].cos())
        if wrapped_av_heading.abs().item() > self.pose_tolerance:
            raise ValueError('AV heading at t0 must be approximately 0 radians.')

        prediction_mask = valid_mask[:, -1]
        prediction_indices = torch.nonzero(prediction_mask, as_tuple=False).flatten()
        if prediction_indices.numel() == 0:
            raise ValueError('No agent is valid at t0, so there is nothing to predict.')
        valid_counts = valid_mask[prediction_indices].sum(dim=-1)
        if (valid_counts < 2).any():
            bad_indices = prediction_indices[valid_counts < 2].tolist()
            raise ValueError(
                'Every predicted agent needs at least two valid historical samples; '
                f'failed agent indices: {bad_indices}.')

        agent_ids = input_data.get('agent_ids')
        prediction_agent_ids = None
        if agent_ids is not None:
            if not isinstance(agent_ids, (list, tuple)) or len(agent_ids) != num_agents:
                raise ValueError(f'agent_ids must be a list/tuple of length {num_agents}.')
            if any(not isinstance(agent_id, str) for agent_id in agent_ids):
                raise ValueError('Every agent_ids entry must be a string.')
            prediction_agent_ids = [agent_ids[index] for index in prediction_indices.tolist()]

        # The decoder only checks whether an agent has any enabled prediction step.
        # Mark t0 for agents that are valid now; history remains solely in valid_mask.
        predict_mask = torch.zeros_like(valid_mask)
        predict_mask[prediction_indices, -1] = True

        data = HeteroData()
        data['agent']['num_nodes'] = num_agents
        data['agent']['position'] = position
        data['agent']['velocity'] = velocity
        data['agent']['heading'] = heading
        data['agent']['valid_mask'] = valid_mask
        data['agent']['predict_mask'] = predict_mask
        data['agent']['type'] = agent_type
        data['agent']['av_index'] = av_index
        data.coordinate_mode = 'ego'
        data.input_coordinate = 'ego'
        data.frequency_hz = frequency_hz
        return AdaptedInput(data, prediction_indices, prediction_agent_ids)
