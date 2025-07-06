from typing import Dict, Any
import torch

class DeepfakeModel:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = None
    def load(self):
        raise NotImplementedError
    def predict(self, frames) -> float:
        raise NotImplementedError
    def warmup(self):
        pass

from .efficientnet import EfficientNetModel

def warmup_models(device: str = None) -> Dict[str, DeepfakeModel]:
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {
        'efficientnet': EfficientNetModel(device),
    }
    for m in models.values():
        m.load()
        m.warmup()
    return models 