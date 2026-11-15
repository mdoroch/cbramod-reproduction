import random

import numpy as np
import torch
import yaml


def load_config(model, overrides):
    """configs/{model}.yaml, then `key=value` overrides (values parsed as YAML)."""
    with open(f"configs/{model}.yaml") as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        if key not in cfg:
            raise KeyError(f"unknown config key: {key}")
        cfg[key] = yaml.safe_load(value)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA / MPS and DataLoader shuffling


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
