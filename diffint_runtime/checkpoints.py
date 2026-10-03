"""Restricted loading of legacy DiffInt tensor checkpoints."""
from argparse import Namespace
from pathlib import Path, PosixPath, WindowsPath
import torch


def load_checkpoint(path, map_location="cpu"):
    # Namespace is present in the author's configuration; do not allow arbitrary classes.
    with torch.serialization.safe_globals([Namespace, Path, PosixPath, WindowsPath]):
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Expected a Lightning tensor state_dict checkpoint")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor)
               for k, v in checkpoint["state_dict"].items()):
        raise ValueError("state_dict must contain only named tensors")
    return checkpoint
