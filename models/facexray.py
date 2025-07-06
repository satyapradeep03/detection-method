from . import DeepfakeModel
from loguru import logger
import torch
import numpy as np

class FaceXRayModel(DeepfakeModel):
    def load(self):
        try:
            from facexray import FaceXRay
            self.model = FaceXRay().to(self.device)
            self.model.eval()
            logger.info("FaceXRay model loaded on {}", self.device)
        except Exception as e:
            logger.exception("Failed to load FaceXRay model: {}", e)
            self.model = None
    def warmup(self):
        if self.model is not None:
            dummy = torch.zeros((1, 3, 224, 224), device=self.device)
            with torch.no_grad():
                _ = self.model(dummy)
    def predict(self, frames):
        if self.model is None:
            return 0.0
        preds = []
        with torch.no_grad():
            for frame in frames:
                img = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).float().to(self.device)
                img = torch.nn.functional.interpolate(img, size=(224,224))
                out = self.model(img)
                prob = float(torch.sigmoid(out).cpu().numpy().flatten()[0])
                preds.append(prob)
        return float(np.mean(preds)) if preds else 0.0 