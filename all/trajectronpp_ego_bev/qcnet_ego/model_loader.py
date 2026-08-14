"""Strict weights-only loader for the deployed Ego-coordinate QCNet."""

from pathlib import Path
from typing import Dict, Tuple

import torch

from .models import QCNetInferenceModel


REQUIRED_CONFIGURATION = {
    'dataset': 'argoverse_v2',
    'input_dim': 2,
    'output_dim': 2,
    'output_head': False,
    'num_historical_steps': 50,
    'num_future_steps': 60,
    'prediction_horizon_steps': 20,
    'trajectory_frequency_hz': 10.0,
    'num_modes': 6,
    'num_recurrent_steps': 3,
    'use_ego_relative': True,
    'use_map': False,
    'coordinate_mode': 'ego',
}

MODEL_CONFIGURATION_KEYS = (
    'dataset', 'input_dim', 'hidden_dim', 'output_dim', 'output_head',
    'num_historical_steps', 'num_future_steps', 'prediction_horizon_steps',
    'trajectory_frequency_hz', 'num_modes', 'num_recurrent_steps',
    'num_freq_bands', 'num_map_layers', 'num_agent_layers', 'num_dec_layers',
    'num_heads', 'head_dim', 'dropout', 'pl2pl_radius', 'time_span',
    'pl2a_radius', 'a2a_radius', 'num_t2m_steps', 'pl2m_radius',
    'a2m_radius', 'use_ego_relative', 'use_map', 'coordinate_mode',
)


def _validate_checkpoint_configuration(hparams: Dict[str, object]) -> None:
    for key, expected in REQUIRED_CONFIGURATION.items():
        if key not in hparams:
            raise RuntimeError(f'Checkpoint is missing required configuration: {key}')
        actual = hparams[key]
        if isinstance(expected, float):
            matches = abs(float(actual) - expected) <= 1e-9
        else:
            matches = actual == expected
        if not matches:
            raise RuntimeError(
                f'Checkpoint configuration mismatch for {key}: '
                f'expected {expected!r}, got {actual!r}.')


def load_inference_model(checkpoint_path: str, device: str) -> Tuple[QCNetInferenceModel, Dict[str, object]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    requested_device = torch.device(device)
    if requested_device.type not in ('cpu', 'cuda'):
        raise ValueError("device must be 'cpu', 'cuda', or a CUDA device such as 'cuda:0'.")
    if requested_device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested, but torch.cuda.is_available() is False.')

    # Only model metadata and state_dict are consumed; unrelated fields are ignored.
    checkpoint = torch.load(path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise RuntimeError('Checkpoint must be a dictionary containing state_dict.')
    hparams = checkpoint.get('hyper_parameters')
    if not isinstance(hparams, dict):
        raise RuntimeError('Checkpoint hyper_parameters must be a dictionary.')
    _validate_checkpoint_configuration(hparams)
    missing_config = [key for key in MODEL_CONFIGURATION_KEYS if key not in hparams]
    if missing_config:
        raise RuntimeError(f'Checkpoint lacks model construction fields: {missing_config}')
    model_config = {key: hparams[key] for key in MODEL_CONFIGURATION_KEYS}

    model = QCNetInferenceModel(model_config)
    try:
        incompatible = model.load_state_dict(checkpoint['state_dict'], strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            'Strict checkpoint loading failed. Decoder shape mismatch and silently '
            'ignored parameters are not allowed.') from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f'Unexpected strict-load result: missing={incompatible.missing_keys}, '
            f'unexpected={incompatible.unexpected_keys}')

    loaded_state = model.state_dict()
    mismatched_values = [
        key for key, value in checkpoint['state_dict'].items()
        if key not in loaded_state or not torch.equal(loaded_state[key].cpu(), value.detach().cpu())
    ]
    if mismatched_values:
        raise RuntimeError(f'Loaded model values do not exactly match checkpoint: {mismatched_values}')
    ego_keys = [
        key for key in loaded_state
        if key == 'encoder.agent_encoder.ego_gate'
        or key.startswith('encoder.agent_encoder.ego_relative_emb.')
    ]
    if not ego_keys or any(key not in checkpoint['state_dict'] for key in ego_keys):
        raise RuntimeError('ego_relative_emb or ego_gate is absent from the checkpoint.')

    model.to(requested_device)
    model.eval()
    audit = {
        'checkpoint_path': str(path),
        'device': str(requested_device),
        'state_tensor_count': len(loaded_state),
        'ego_tensor_count': len(ego_keys),
        'missing_keys': [],
        'unexpected_keys': [],
        'strict': True,
        'weights_only_initialization': True,
        'training_state_restored': False,
        'configuration': model_config,
    }
    return model, audit
