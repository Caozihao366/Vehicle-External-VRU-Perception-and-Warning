"""Pure PyTorch QCNet graph used only for deployment inference."""

from typing import Mapping

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from .qcnet_decoder import QCNetDecoder
from .qcnet_encoder import QCNetEncoder


class QCNetInferenceModel(nn.Module):
    """QCNet encoder/decoder without Lightning, losses, metrics, or training hooks."""

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__()
        self.num_historical_steps = int(config['num_historical_steps'])
        self.num_future_steps = int(config['num_future_steps'])
        self.prediction_horizon_steps = int(config['prediction_horizon_steps'])
        self.trajectory_frequency_hz = float(config['trajectory_frequency_hz'])
        self.num_modes = int(config['num_modes'])
        self.output_dim = int(config['output_dim'])
        self.use_ego_relative = bool(config['use_ego_relative'])
        self.use_map = bool(config['use_map'])

        self.encoder = QCNetEncoder(
            dataset=str(config['dataset']),
            input_dim=int(config['input_dim']),
            hidden_dim=int(config['hidden_dim']),
            num_historical_steps=self.num_historical_steps,
            pl2pl_radius=float(config['pl2pl_radius']),
            time_span=int(config['time_span']),
            pl2a_radius=float(config['pl2a_radius']),
            a2a_radius=float(config['a2a_radius']),
            num_freq_bands=int(config['num_freq_bands']),
            num_map_layers=int(config['num_map_layers']),
            num_agent_layers=int(config['num_agent_layers']),
            num_heads=int(config['num_heads']),
            head_dim=int(config['head_dim']),
            dropout=float(config['dropout']),
            use_ego_relative=self.use_ego_relative,
            use_map=self.use_map,
        )
        self.decoder = QCNetDecoder(
            dataset=str(config['dataset']),
            input_dim=int(config['input_dim']),
            hidden_dim=int(config['hidden_dim']),
            output_dim=self.output_dim,
            output_head=bool(config['output_head']),
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            num_modes=self.num_modes,
            num_recurrent_steps=int(config['num_recurrent_steps']),
            num_t2m_steps=int(config['num_t2m_steps']),
            pl2m_radius=float(config['pl2m_radius']),
            a2m_radius=float(config['a2m_radius']),
            num_freq_bands=int(config['num_freq_bands']),
            num_layers=int(config['num_dec_layers']),
            num_heads=int(config['num_heads']),
            head_dim=int(config['head_dim']),
            dropout=float(config['dropout']),
            use_map=self.use_map,
        )

    def forward(self, data: HeteroData) -> Mapping[str, torch.Tensor]:
        scene_encoding = self.encoder(data)
        return self.decoder(data, scene_encoding)
