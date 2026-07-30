# -*- coding: utf-8 -*-
"""Ego-BEV Trajectron++ inference wrapper for upstream BEV detections.

Expected input frame (must match training):
  - origin: current ego pose
  - x: forward (m), y: left (m)
  - heading: radians, 0 along +x, CCW positive
  - dt: 0.5 s (2 Hz)
"""
from __future__ import print_function

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_CODE_ROOT = os.path.join(_PKG_ROOT, 'code', 'trajectron')
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)

from environment import Environment, Scene, Node, derivative_of
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron


STANDARDIZATION = {
    'PEDESTRIAN': {
        'position': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}},
        'velocity': {'x': {'mean': 0, 'std': 2}, 'y': {'mean': 0, 'std': 2}},
        'acceleration': {'x': {'mean': 0, 'std': 1}, 'y': {'mean': 0, 'std': 1}},
    },
    'VEHICLE': {
        'position': {'x': {'mean': 0, 'std': 80}, 'y': {'mean': 0, 'std': 80}},
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15},
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4},
        },
        'heading': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1},
            u'\xb0': {'mean': 0, 'std': np.pi},
            u'd\xb0': {'mean': 0, 'std': 1},
        },
    },
}


def _vehicle_columns():
    cols = pd.MultiIndex.from_product(
        [['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
    cols = cols.append(pd.MultiIndex.from_tuples([(u'heading', u'\xb0'), (u'heading', u'd\xb0')]))
    cols = cols.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))
    return cols


def _pedestrian_columns():
    return pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])


def _infer_heading(xy, dt):
    """Fallback heading from velocity when upstream does not provide one."""
    vx = derivative_of(xy[:, 0], dt)
    vy = derivative_of(xy[:, 1], dt)
    heading = np.arctan2(vy, vx)
    heading[~np.isfinite(heading)] = 0.0
    return heading


def _build_vehicle_df(xy, heading, dt):
    x = xy[:, 0].astype(float)
    y = xy[:, 1].astype(float)
    vx = derivative_of(x, dt)
    vy = derivative_of(y, dt)
    ax = derivative_of(vx, dt)
    ay = derivative_of(vy, dt)
    heading = np.asarray(heading, dtype=float)
    dheading = derivative_of(heading, dt, radian=True)
    data = np.stack([
        x, y, vx, vy, ax, ay,
        np.cos(heading), np.sin(heading),
        heading, dheading,
        np.linalg.norm(np.stack([vx, vy], axis=-1), axis=-1),
        np.linalg.norm(np.stack([ax, ay], axis=-1), axis=-1),
    ], axis=-1)
    return pd.DataFrame(data, columns=_vehicle_columns())


def _build_pedestrian_df(xy, dt):
    x = xy[:, 0].astype(float)
    y = xy[:, 1].astype(float)
    vx = derivative_of(x, dt)
    vy = derivative_of(y, dt)
    ax = derivative_of(vx, dt)
    ay = derivative_of(vy, dt)
    data = np.stack([x, y, vx, vy, ax, ay], axis=-1)
    return pd.DataFrame(data, columns=_pedestrian_columns())


class EgoBevPredictor(object):
    """Load int_ee_ego_bev weights and predict futures from ego-BEV tracks."""

    def __init__(self, package_root=None, checkpoint=200, device='cpu'):
        self.package_root = os.path.abspath(package_root or _PKG_ROOT)
        self.weights_dir = os.path.join(self.package_root, 'weights')
        self.device = device
        self.checkpoint = checkpoint

        with open(os.path.join(self.weights_dir, 'config.json'), 'r') as f:
            self.hyperparams = json.load(f)

        self.dt = 0.5
        # Deploy default: 1.5s = 3 steps (model was trained with ph=6 / 3s).
        self.ph = 3
        self.train_ph = int(self.hyperparams['prediction_horizon'])
        self.max_hl = int(self.hyperparams['maximum_history_length'])

        self.env = Environment(
            node_type_list=['VEHICLE', 'PEDESTRIAN'],
            standardization=STANDARDIZATION,
        )
        self.env.attention_radius = {
            (self.env.NodeType.PEDESTRIAN, self.env.NodeType.PEDESTRIAN): 10.0,
            (self.env.NodeType.PEDESTRIAN, self.env.NodeType.VEHICLE): 20.0,
            (self.env.NodeType.VEHICLE, self.env.NodeType.PEDESTRIAN): 20.0,
            (self.env.NodeType.VEHICLE, self.env.NodeType.VEHICLE): 30.0,
        }
        self.env.robot_type = self.env.NodeType.VEHICLE
        # Trajectron reads dt from env.scenes[0] during model init.
        self.env.scenes = [Scene(timesteps=1, dt=self.dt, name='bootstrap')]

        self.model_registrar = ModelRegistrar(self.weights_dir, self.device)
        self.model_registrar.load_models(self.checkpoint)
        self.trajectron = Trajectron(self.model_registrar, self.hyperparams, None, self.device)
        self.trajectron.set_environment(self.env)
        self.trajectron.set_annealing_params()

    def predict(self, agents, prediction_horizon=None, prediction_horizon_sec=None,
                num_samples=1, z_mode=True, gmm_mode=True):
        """Predict futures for one ego-BEV frame history.

        Args:
            agents: list of dicts, each with:
              - id: str
              - type: 'VEHICLE' or 'PEDESTRIAN'
              - history_xy: array (T, 2) in current ego BEV, oldest -> newest
              - history_heading: optional array (T,) radians (VEHICLE recommended)
              - is_ego: optional bool
            prediction_horizon: steps to predict (default 3 => 1.5s at dt=0.5).
            prediction_horizon_sec: optional seconds; overrides prediction_horizon when set.
            num_samples: multimodal samples (1 with z_mode/gmm_mode = most-likely)
            z_mode / gmm_mode: most-likely latent / GMM mode

        Returns:
            dict: agent_id -> ndarray (num_samples, prediction_horizon, 2)
                  absolute ego-BEV positions in meters.
                  Default shape last dim time: 3 steps @ 0.5/1.0/1.5 s.
        """
        if not agents:
            return {}

        if prediction_horizon_sec is not None:
            ph = int(round(float(prediction_horizon_sec) / self.dt))
        elif prediction_horizon is not None:
            ph = int(prediction_horizon)
        else:
            ph = int(self.ph)
        if ph < 1:
            raise ValueError('prediction horizon must be >= 1 step')

        hist_lens = [np.asarray(a['history_xy']).shape[0] for a in agents]
        T = int(max(hist_lens))
        if T < 2:
            raise ValueError('Need at least 2 history timesteps to estimate velocity/acceleration.')

        scene = Scene(timesteps=T, dt=self.dt, name='online_ego_bev', aug_func=None)
        nodes = []
        for agent in agents:
            agent_type = str(agent['type']).upper()
            xy = np.asarray(agent['history_xy'], dtype=float)
            if xy.ndim != 2 or xy.shape[1] != 2:
                raise ValueError('history_xy must be (T, 2) for agent %s' % agent['id'])
            if xy.shape[0] < T:
                pad = np.full((T - xy.shape[0], 2), np.nan, dtype=float)
                xy = np.vstack([pad, xy])

            first_valid = int(np.where(~np.isnan(xy[:, 0]))[0][0])
            xy_valid = xy[first_valid:]

            if agent_type == 'VEHICLE':
                if agent.get('history_heading') is not None:
                    heading = np.asarray(agent['history_heading'], dtype=float)
                    if heading.shape[0] < T:
                        heading = np.concatenate([np.full(T - heading.shape[0], np.nan), heading])
                    heading = heading[first_valid:]
                else:
                    heading = _infer_heading(xy_valid, self.dt)
                df = _build_vehicle_df(xy_valid, heading, self.dt)
                node_type = self.env.NodeType.VEHICLE
            elif agent_type == 'PEDESTRIAN':
                df = _build_pedestrian_df(xy_valid, self.dt)
                node_type = self.env.NodeType.PEDESTRIAN
            else:
                raise ValueError('Unsupported type %s' % agent_type)

            node = Node(
                node_type=node_type,
                node_id=str(agent['id']),
                data=df,
                first_timestep=first_valid,
                is_robot=bool(agent.get('is_ego', False)),
            )
            nodes.append(node)
            if node.is_robot:
                scene.robot = node

        scene.nodes = nodes
        scene.calculate_scene_graph(
            self.env.attention_radius,
            self.hyperparams['edge_addition_filter'],
            self.hyperparams['edge_removal_filter'],
        )

        # Temporarily attach this scene so Trajectron can read env metadata.
        old_scenes = self.env.scenes
        self.env.scenes = [scene]
        self.trajectron.set_environment(self.env)

        t = T - 1
        with torch.no_grad():
            preds = self.trajectron.predict(
                scene,
                np.array([t]),
                ph,
                num_samples=num_samples,
                min_future_timesteps=0,
                z_mode=z_mode,
                gmm_mode=gmm_mode,
                full_dist=False,
            )

        self.env.scenes = old_scenes

        out = {}
        if t not in preds:
            return out
        for node, traj in preds[t].items():
            arr = np.asarray(traj)
            # Trajectron returns (samples, 1, ph, 2) when predicting one timestep.
            if arr.ndim == 4 and arr.shape[1] == 1:
                arr = arr[:, 0]
            out[node.id] = arr
        return out
