"""
Lightweight persistent store for named objects (reference images + embeddings).

Layout:
  {base_dir}/objects/{safe_label}/meta.json     — label + description
  {base_dir}/objects/{safe_label}/ref_image.jpg — reference crop (BGR)
  {base_dir}/objects/{safe_label}/ref_emb.npy   — EfficientNetLite embedding (float32)
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class ObjectStore:
    def __init__(self, base_dir: str):
        self._root = Path(base_dir) / "objects"
        self._root.mkdir(parents=True, exist_ok=True)

    def _safe(self, label: str) -> str:
        return re.sub(r"[^\w\-:.]+", "_", label.strip()) or "default"

    def _dir(self, label: str) -> Path:
        d = self._root / self._safe(label)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(
        self,
        label: str,
        description: str,
        image: np.ndarray,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        d = self._dir(label)
        meta = {
            "label": label,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(d / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        cv2.imwrite(str(d / "ref_image.jpg"), image)
        if embedding is not None:
            np.save(str(d / "ref_emb.npy"), embedding.astype(np.float32))

    def load(self, label: str) -> Optional[dict]:
        d = self._root / self._safe(label)
        meta_path = d / "meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        image = None
        img_path = d / "ref_image.jpg"
        if img_path.exists():
            image = cv2.imread(str(img_path))
        embedding = None
        emb_path = d / "ref_emb.npy"
        if emb_path.exists():
            embedding = np.load(str(emb_path))
        return {
            "label": meta["label"],
            "description": meta.get("description", ""),
            "image": image,
            "embedding": embedding,
        }

    _LABEL_CONFIDENCE = 0.6

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Text search over label and description; no ML required."""
        q = query.lower()
        q_words = set(q.split())
        results = []
        for d in self._root.iterdir():
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue
            label = meta.get("label", "")
            desc = meta.get("description", "")
            label_l = label.lower()
            desc_l = desc.lower()

            if q in label_l:
                label_score = 1.0
            else:
                label_words = set(label_l.split())
                overlap = q_words & label_words
                label_score = len(overlap) / max(len(q_words), len(label_words), 1)

            if q in desc_l:
                desc_score = 1.0
            else:
                desc_words = set(desc_l.split())
                overlap = q_words & desc_words
                desc_score = len(overlap) / max(len(q_words), len(desc_words), 1)

            if label_score >= self._LABEL_CONFIDENCE:
                score = label_score
            else:
                score = (label_score + desc_score) / 2

            if score > 0:
                results.append({"label": label, "description": desc, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def list_labels(self) -> list[str]:
        labels = []
        for d in self._root.iterdir():
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    labels.append(json.load(f).get("label", d.name))
            except Exception:
                pass
        return sorted(labels)
