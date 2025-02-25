dependencies = ['torch', 'torchvision']

import torch
from megaloc_model import MegaLocModel


def get_trained_model() -> torch.nn.Module:
    model = MegaLocModel()
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://github.com/gmberton/MegaLoc/releases/download/v1.0/megaloc.torch",
        map_location=torch.device('cpu'))
    )
    return model
