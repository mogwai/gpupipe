"""PyTorch is optional.

CPU pipelines run without it, and their worker processes never import it
(importing torch costs ~0.4s and ~170 MB per process). Torch features switch
on when it is there: tensor fast paths once a stage has imported torch, GPU
pinning/release when a GPU stage is added.
"""
import sys


def loaded():
    """torch if this process has already imported it (a stage uses it), else
    None. Never imports it."""
    return sys.modules.get("torch")


def available():
    """torch, importing it if installed; None if it isn't."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def required(what):
    """torch, or an ImportError saying `what` needs it."""
    torch = available()
    if torch is None:
        raise ImportError(f"{what} needs PyTorch: pip install 'gpupipe[torch]' (or install torch)")
    return torch
