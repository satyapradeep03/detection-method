from . import DeepfakeModel
from loguru import logger
import torch
import numpy as np

class EfficientNetModel(DeepfakeModel):
    def load(self):
        try:
            from efficientnet_pytorch import EfficientNet
            self.model = EfficientNet.from_pretrained('efficientnet-b4').to(self.device)
            self.model.eval()
            logger.info("EfficientNet-B4 loaded on {}", self.device)
        except Exception as e:
            logger.exception("Failed to load EfficientNet-B4: {}", e)
            self.model = None
    def warmup(self):
        if self.model is not None:
            dummy = torch.zeros((1, 3, 380, 380), device=self.device)
            with torch.no_grad():
                _ = self.model(dummy)
    def predict(self, frames):
        if self.model is None:
            return 0.0
        preds = []
        with torch.no_grad():
            for frame in frames:
                img = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float().to(self.device)
                img = torch.nn.functional.interpolate(img, size=(380,380))
                out = self.model(img)
                prob = float(torch.sigmoid(out).cpu().numpy().flatten()[0])
                preds.append(prob)
        return float(np.mean(preds)) if preds else 0.0 