from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from attrs import inspect
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image


@dataclass
class DepthFrame:
    depth_map: np.ndarray                       # HxW float32, best available depth (metric preferred over relative)
    rays: np.ndarray = None                     # HxWx3 float32, unit direction vectors per pixel
    camera_pose: Optional[np.ndarray] = None    # 4x4 float64 extrinsic, None if not estimated
    intrinsics: Optional[np.ndarray] = None     # 3x3 float32 camera intrinsic matrix K
    sky: Optional[np.ndarray] = None            # HxW float32, sky mask if model supports it


class BaseDepthEstimator(ABC):
    @abstractmethod
    def estimate(self, rgb_frame: np.ndarray) -> DepthFrame:
        """Estimate depth from a single RGB frame (HxWx3, uint8)."""
        ...


class DA3Estimator(BaseDepthEstimator):
    """
    Depth Anything 3 — unified depth-ray-camera estimator.
    Uses depth_anything_3.api directly: returns metric depth, actual camera
    intrinsics (K matrix), and global camera pose (c2w) per frame.
    """
    def __init__(self, model_id: str = "depth-anything/da3-large", device: Optional[str] = None):
        try:
            from depth_anything_3.api import DepthAnything3
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = DepthAnything3.from_pretrained(model_id).to(self.device)
            self.model.eval()
        except Exception as e:
            raise ImportError(
                f"Could not load DA3 model '{model_id}'. "
f"Original error: {e}"
            ) from e

    def estimate(self, rgb_frame: np.ndarray) -> DepthFrame:
        return self.estimate_batch([rgb_frame])[0]

    def estimate_batch(self, rgb_frames: list) -> list:
        """
        Run DA3 multi-view inference on all frames at once.
        This is the correct usage: DA3 jointly reasons across all views for
        globally consistent depth and camera poses.
        """
        import cv2

        prediction = self.model.inference(rgb_frames)
        import inspect
        print(inspect.getsource(self.model.forward))
        print(self.model)

        results = []
        for i, rgb_frame in enumerate(rgb_frames):
            h, w = rgb_frame.shape[:2]

            depth_map = prediction.depth[i].astype(np.float32)
            if depth_map.shape != (h, w):
                depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

            if prediction.intrinsics is not None:
                rays = _build_rays_from_K(h, w, prediction.intrinsics[i].astype(np.float32))
            else:
                rays = _build_rays(h, w)

            c2w = None
            if prediction.extrinsics is not None:
                w2c = prediction.extrinsics[i].astype(np.float64)
                if w2c.shape == (3, 4):
                    tmp = np.eye(4, dtype=np.float64)
                    tmp[:3, :] = w2c
                    w2c = tmp
                c2w = np.linalg.inv(w2c)

            K = None
            if prediction.intrinsics is not None:
                K = prediction.intrinsics[i].astype(np.float32)

            results.append(DepthFrame(depth_map=depth_map, rays=rays, camera_pose=c2w, intrinsics=K))

        return results


def _build_rays_from_K(h: int, w: int, K: np.ndarray) -> np.ndarray:
    """Build per-pixel ray directions from an actual 3x3 camera intrinsic matrix."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    return (dirs / norms).astype(np.float32)


def _build_rays(h: int, w: int) -> np.ndarray:
    """Fallback ray builder using a pinhole approximation (fx=fy=0.8*max(W,H))."""
    fx = fy = max(w, h) * 0.8
    cx, cy = w / 2.0, h / 2.0
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    return (dirs / norms).astype(np.float32)


class DA3OnnxEstimator(BaseDepthEstimator):
    """
    ONNX-runtime Depth Anything 3 estimator.
    Supports both DA3-METRIC (outputs metric_depth + sky) and DA3-BASE (depth only).
    """

    def __init__(self, onnx_path: str, device: str = "cpu"):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Read fixed input spatial dims from the model (shape is [B, C, H, W]).
        # Dim values are ints when fixed, strings when dynamic.
        shape = self.session.get_inputs()[0].shape
        self._model_h = shape[2] if isinstance(shape[2], int) else None
        self._model_w = shape[3] if isinstance(shape[3], int) else None

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def estimate(self, rgb_frame: np.ndarray) -> DepthFrame:
        h, w = rgb_frame.shape[:2]
        x = self._preprocess(rgb_frame)
        outputs = dict(zip(self.output_names, self.session.run(self.output_names, {self.input_name: x})))

        depth = self._resize(outputs["depth"][0].astype(np.float32), w, h)

        metric_depth = None
        if "metric_depth" in outputs:
            metric_depth = self._resize(outputs["metric_depth"][0].astype(np.float32), w, h)

        sky = None
        if "sky" in outputs:
            sky = self._resize(outputs["sky"][0].astype(np.float32), w, h)

        return DepthFrame(
            depth_map=metric_depth if metric_depth is not None else depth,
            sky=sky,
            rays=_build_rays(h, w),
        )

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
        if self._model_h is not None and self._model_w is not None:
            if pil.height != self._model_h or pil.width != self._model_w:
                pil = pil.resize((self._model_w, self._model_h), Image.BILINEAR)
        tensor = self.transform(pil).unsqueeze(0)
        return tensor.numpy().astype(np.float32)

    def _resize(self, x: np.ndarray, w: int, h: int) -> np.ndarray:
        # Squeeze any leading size-1 dims so we always work with (H, W)
        while x.ndim > 2 and x.shape[0] == 1:
            x = x[0]
        if x.shape != (h, w):
            x = cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
        return x


def build_estimator(**kwargs) -> BaseDepthEstimator:
    if "onnx_path" in kwargs:
        return DA3OnnxEstimator(onnx_path=kwargs["onnx_path"], device=kwargs.get("device", "cpu"))
    kwargs.pop("prefer_da3", None)
    return DA3Estimator(**kwargs)
