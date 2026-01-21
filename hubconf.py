"""PyTorch Hub configuration for MegaLoc.

Usage:
    import torch
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
"""

dependencies = ["torch", "torchvision", "huggingface_hub"]

import torch  # noqa: E402

from megaloc_model import MegaLoc  # noqa: E402


def get_trained_model() -> torch.nn.Module:
    """Load the pretrained MegaLoc model.

    Returns:
        MegaLoc model with pretrained weights.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    model = MegaLoc()

    weights_path = hf_hub_download(
        repo_id="gberton/MegaLoc",
        filename="model.safetensors",
    )

    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)

    return model
