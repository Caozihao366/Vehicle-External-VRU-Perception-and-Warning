# -*- coding: utf-8 -*-
"""Minimal smoke test: synthetic ego-BEV tracks -> Trajectron++ futures."""
from __future__ import print_function

import os
import sys

import numpy as np

PKG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PKG, 'inference'))

from ego_bev_predictor import EgoBevPredictor


def main():
    dt = 0.5
    T = 9  # 8 history intervals + current

    # Ego stays at origin in current-ego frame (static for this toy example).
    ego_xy = np.zeros((T, 2), dtype=float)
    ego_heading = np.zeros(T, dtype=float)

    # A vehicle ahead moving +x at ~10 m/s in ego BEV.
    t = np.arange(T, dtype=float)
    veh_xy = np.stack([20.0 + 10.0 * (t - (T - 1)) * dt, np.full(T, 3.0)], axis=1)
    veh_heading = np.zeros(T, dtype=float)

    # A pedestrian to the left, nearly static.
    ped_xy = np.stack([8.0 + 0.2 * (t - (T - 1)) * dt, np.full(T, 6.0)], axis=1)

    agents = [
        {'id': 'ego', 'type': 'VEHICLE', 'history_xy': ego_xy,
         'history_heading': ego_heading, 'is_ego': True},
        {'id': 'veh_1', 'type': 'VEHICLE', 'history_xy': veh_xy,
         'history_heading': veh_heading},
        {'id': 'ped_1', 'type': 'PEDESTRIAN', 'history_xy': ped_xy},
    ]

    predictor = EgoBevPredictor(package_root=PKG, checkpoint=200, device='cpu')
    preds = predictor.predict(agents, prediction_horizon_sec=1.5,
                              num_samples=1, z_mode=True, gmm_mode=True)

    print('horizon_default_steps', predictor.ph, '=>', predictor.ph * predictor.dt, 's')
    print('predicted agents:', sorted(preds.keys()))
    for aid, traj in preds.items():
        print(aid, 'shape', traj.shape, 't=0.5', traj[0, 0], 't=1.5', traj[0, -1])


if __name__ == '__main__':
    main()
