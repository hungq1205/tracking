import torch
import numpy as np


class EfficientNetLiteEmbedder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_embedding(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return torch.randn(1, 1280)
