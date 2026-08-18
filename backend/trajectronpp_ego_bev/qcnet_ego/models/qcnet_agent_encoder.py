# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import subgraph

from ..modules import AttentionLayer
from ..modules import FourierEmbedding
from ..modules import angle_between_2d_vectors
from ..modules import weight_init
from ..modules import wrap_angle


class QCNetAgentEncoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 use_ego_relative: bool = False,
                 use_map: bool = True) -> None:
        super(QCNetAgentEncoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.use_ego_relative = use_ego_relative
        self.use_map = use_map

        if dataset == 'argoverse_v2':
            input_dim_x_a = 4
            input_dim_r_t = 4
            input_dim_r_pl2a = 3
            input_dim_r_a2a = 3
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))

        if dataset == 'argoverse_v2':
            self.type_a_emb = nn.Embedding(10, hidden_dim)
        else:
            raise ValueError('{} is not a valid dataset'.format(dataset))
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        # Ego-relative inputs are [x_rel, y_rel, vx_rel, vy_rel, heading_rel].
        # This is a separate branch so the pretrained x_a_emb parameter shapes remain unchanged.
        self.ego_relative_emb = (
            FourierEmbedding(input_dim=5, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
            if use_ego_relative else None
        )
        self.ego_gate = nn.Parameter(torch.empty(())) if use_ego_relative else None
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.apply(weight_init)
        if self.ego_gate is not None:
            nn.init.constant_(self.ego_gate, 0.01)

    def forward(self,
                data: HeteroData,
                map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        pos_a = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].contiguous()
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)
        head_a = data['agent']['heading'][:, :self.num_historical_steps].contiguous()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        if self.use_map:
            pos_pl = data['map_polygon']['position'][:, :self.input_dim].contiguous()
            orient_pl = data['map_polygon']['orientation'].contiguous()
        if self.dataset == 'argoverse_v2':
            vel = data['agent']['velocity'][:, :self.num_historical_steps, :self.input_dim].contiguous()
            length = width = height = None
            categorical_embs = [
                self.type_a_emb(data['agent']['type'].long()).repeat_interleave(repeats=self.num_historical_steps,
                                                                                dim=0),
            ]
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

        if self.dataset == 'argoverse_v2':
            x_a = torch.stack(
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
                 torch.norm(vel[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2])], dim=-1)
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs)
        x_a = x_a.view(-1, self.num_historical_steps, self.hidden_dim)
        if self.use_ego_relative:
            # av_index identifies the AV node in each scene. In a PyG Batch, QCNet's
            # train/val/test steps convert these indices to global node indices before
            # forward; for a single HeteroData object the index is already local/global.
            av_index = torch.as_tensor(data['agent']['av_index'], device=pos_a.device).reshape(-1).long()
            ego_pos = pos_a[av_index, :, :2]
            ego_head = head_a[av_index]
            ego_vel = vel[av_index, :, :2]
            ego_mask = mask[av_index]
            if isinstance(data, Batch):
                agent_batch = data['agent']['batch'].long()
                ego_pos = ego_pos[agent_batch]
                ego_head = ego_head[agent_batch]
                ego_vel = ego_vel[agent_batch]
                ego_mask = ego_mask[agent_batch]
            else:
                num_agents = pos_a.size(0)
                ego_pos = ego_pos.expand(num_agents, -1, -1)
                ego_head = ego_head.expand(num_agents, -1)
                ego_vel = ego_vel.expand(num_agents, -1, -1)
                ego_mask = ego_mask.expand(num_agents, -1)

            # Rotate global differences by -ego_head at every historical timestep:
            # x_rel = cos(theta) * dx + sin(theta) * dy
            # y_rel = -sin(theta) * dx + cos(theta) * dy.
            rel_pos_global = pos_a[:, :, :2] - ego_pos
            rel_vel_global = vel[:, :, :2] - ego_vel
            cos_ego, sin_ego = ego_head.cos(), ego_head.sin()
            rel_pos = torch.stack(
                [cos_ego * rel_pos_global[..., 0] + sin_ego * rel_pos_global[..., 1],
                 -sin_ego * rel_pos_global[..., 0] + cos_ego * rel_pos_global[..., 1]], dim=-1)
            rel_vel = torch.stack(
                [cos_ego * rel_vel_global[..., 0] + sin_ego * rel_vel_global[..., 1],
                 -sin_ego * rel_vel_global[..., 0] + cos_ego * rel_vel_global[..., 1]], dim=-1)
            rel_head = wrap_angle(head_a - ego_head)
            ego_relative = torch.cat([rel_pos, rel_vel, rel_head.unsqueeze(-1)], dim=-1)

            # An ego-relative state is meaningful only when both the agent and AV
            # are valid. Mask again after embedding because a zero Fourier input
            # can still produce a nonzero learned embedding.
            ego_relative_mask = mask & ego_mask
            ego_relative = ego_relative.masked_fill(~ego_relative_mask.unsqueeze(-1), 0.0)
            ego_feature = self.ego_relative_emb(
                continuous_inputs=ego_relative.view(-1, ego_relative.size(-1)),
                categorical_embs=None)
            ego_feature = ego_feature.view(-1, self.num_historical_steps, self.hidden_dim)
            ego_feature = ego_feature * ego_relative_mask.unsqueeze(-1)
            x_a = x_a + self.ego_gate * ego_feature

        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
             rel_head_t,
             edge_index_t[0] - edge_index_t[1]], dim=-1)
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        mask_s = mask.transpose(0, 1).reshape(-1)
        if self.use_map:
            pos_pl = pos_pl.repeat(self.num_historical_steps, 1)
            orient_pl = orient_pl.repeat(self.num_historical_steps)
        if isinstance(data, Batch):
            batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                 for t in range(self.num_historical_steps)], dim=0)
            if self.use_map:
                batch_pl = torch.cat([data['map_polygon']['batch'] + data.num_graphs * t
                                      for t in range(self.num_historical_steps)], dim=0)
        else:
            batch_s = torch.arange(self.num_historical_steps,
                                   device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
            if self.use_map:
                batch_pl = torch.arange(self.num_historical_steps,
                                        device=pos_pl.device).repeat_interleave(data['map_polygon']['num_nodes'])
        if self.use_map:
            edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius,
                                     batch_x=batch_s, batch_y=batch_pl, max_num_neighbors=300)
            edge_index_pl2a = edge_index_pl2a[:, mask_s[edge_index_pl2a[1]]]
            rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
            rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
            r_pl2a = torch.stack(
                [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]],
                                          nbr_vector=rel_pos_pl2a[:, :2]),
                 rel_orient_pl2a], dim=-1)
            r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        edge_index_a2a = radius_graph(x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False,
                                      max_num_neighbors=300)
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
             rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        for i in range(self.num_layers):
            x_a = x_a.reshape(-1, self.hidden_dim)
            x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t)
            x_a = x_a.reshape(-1, self.num_historical_steps,
                              self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            if self.use_map:
                x_a = self.pl2a_attn_layers[i](
                    (map_enc['x_pl'].transpose(0, 1).reshape(-1, self.hidden_dim), x_a),
                    r_pl2a, edge_index_pl2a)
            x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)
            x_a = x_a.reshape(self.num_historical_steps, -1, self.hidden_dim).transpose(0, 1)

        return {'x_a': x_a}
