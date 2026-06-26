#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch import nn

import onnx
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as T
import numpy as np

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

PATCH_SIZE = 14


# =========================
# ONNX WRAPPER (FIXED)
# =========================
class DepthAnything3OnnxWrapper(nn.Module):
    def __init__(self, api_model: DepthAnything3):
        super().__init__()
        self.model = api_model

    def forward(self, image: torch.Tensor):
        # (B,3,H,W) → model expects (B,1,3,H,W)
        model_in = image.unsqueeze(1)

        output = self.model(
            model_in,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=[],
            infer_gs=False,
        )

        depth = output["depth"]

        # SAFE: sky may not exist in some checkpoints
        sky = output.get("sky", torch.zeros_like(depth))

        return depth, sky


# =========================
# ARGS
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, default="DA3METRIC-LARGE")
    p.add_argument("--onnx-path", type=str, default=None)
    p.add_argument("--height", type=int, default=280)
    p.add_argument("--width", type=int, default=504)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--opset", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default=".")
    p.add_argument("--demo-image", type=str, default="assets/examples/ika/0006.jpg")
    return p.parse_args()


# =========================
# LOAD MODEL
# =========================
def load_model(model_dir, device):
    model = DepthAnything3.from_pretrained(model_dir).to(device)
    model.eval()
    return model


# =========================
# EXPORT ONNX
# =========================
def export_onnx(model_dir, onnx_path, h, w, opset, device):
    if h % PATCH_SIZE != 0 or w % PATCH_SIZE != 0:
        raise ValueError("H and W must be divisible by 14")

    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    model = load_model(Path(model_dir), device)
    wrapper = DepthAnything3OnnxWrapper(model).to(device)

    dummy = torch.zeros(1, 3, h, w, device=device)

    print("Running warmup forward...")
    with torch.no_grad():
        _ = wrapper(dummy)

    print("Exporting ONNX...")

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path.as_posix(),
        input_names=["image"],
        output_names=["depth", "sky"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={
            "image": {0: "batch"},
            "depth": {0: "batch"},
            "sky": {0: "batch"},
        },
    )

    print(f"[OK] ONNX saved to {onnx_path}")


# =========================
# VERIFY
# =========================
def verify(onnx_path):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    for inp in model.graph.input:
        print("INPUT:", inp.name)

    for out in model.graph.output:
        print("OUTPUT:", out.name)


# =========================
# DEMO RUN
# =========================
def run_demo(onnx_path, image_path):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    inp = sess.get_inputs()[0]
    h, w = inp.shape[2], inp.shape[3]

    img = Image.open(image_path).convert("RGB").resize((w, h))
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    x = tf(img).unsqueeze(0).numpy().astype(np.float32)

    depth, sky = sess.run(None, {"image": x})

    depth = depth[0]

    print("[DEMO] depth:", depth.shape, depth.min(), depth.max())
    print("[DEMO] sky:", sky.shape, sky.min(), sky.max())

    vis = visualize_depth(depth, ret_type=np.uint8)
    Image.fromarray(vis).save("depth_vis.png")


# =========================
# MAIN
# =========================
def main():
    args = parse_args()

    model_name = Path(args.model_dir).name
    onnx_path = Path(args.onnx_path) if args.onnx_path else Path(args.output_dir) / f"{model_name}.onnx"

    export_onnx(
        args.model_dir,
        onnx_path,
        args.height,
        args.width,
        args.opset,
        torch.device(args.device),
    )

    verify(onnx_path)

    if args.demo_image:
        run_demo(onnx_path, args.demo_image)


if __name__ == "__main__":
    main()