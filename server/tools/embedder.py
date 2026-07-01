import cv2
import torch
from transformers import AutoImageProcessor, AutoModel

_MODEL_ID = "facebook/dinov2-small"


class DINOv2Embedder:
    """Re-ID embeddings via DINOv2 ViT-S/14 (384-dim), used for cosine-similarity
    matching of tracked/remembered objects (cosine >= 0.75 = same target)."""

    def __init__(self, model_id: str = _MODEL_ID):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    @torch.inference_mode()
    def get_embedding(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=crop_rgb, return_tensors="pt").to(self.device)
        embedding = self.model(**inputs).pooler_output  # (1, 384)
        return embedding.detach().cpu()
