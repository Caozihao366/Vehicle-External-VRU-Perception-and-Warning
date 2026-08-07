"""Command-line entry point for the split TTC package."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .adapter import compute_multimodal_ttc_from_payload
from .core import compute_ttc
from .models import EgoState, VRUState


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal TTC baseline.")
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Multimodal prediction payload; when set, other scenario flags are ignored",
    )
    parser.add_argument("--ego-speed", type=float, default=5.0, help="m/s")
    parser.add_argument("--ego-yaw-rate", type=float, default=0.0, help="rad/s")
    parser.add_argument("--vru-x", type=float, default=15.0, help="metres forward")
    parser.add_argument("--vru-y", type=float, default=0.0, help="metres left")
    parser.add_argument("--vru-vx", type=float, default=0.0, help="m/s forward")
    parser.add_argument("--vru-vy", type=float, default=0.0, help="m/s left")
    parser.add_argument("--horizon", type=float, default=5.0, help="seconds")
    parser.add_argument("--dt", type=float, default=0.1, help="seconds")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.input_json is not None:
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        result = compute_multimodal_ttc_from_payload(payload)
        print(json.dumps(asdict(result), indent=2, allow_nan=False))
        return

    result = compute_ttc(
        EgoState(speed_mps=args.ego_speed, yaw_rate_rps=args.ego_yaw_rate),
        VRUState(
            x=args.vru_x,
            y=args.vru_y,
            vx_mps=args.vru_vx,
            vy_mps=args.vru_vy,
        ),
        horizon_s=args.horizon,
        dt_s=args.dt,
    )
    print(json.dumps(asdict(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
