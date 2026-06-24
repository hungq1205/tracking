import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

CHUNK_SIZE_WORDS = 200
EMBED_DIM = 384
CLIP_EMBED_DIM = 512


class RagStore:
    """
    Semantic memory store backed by sentence-transformers embeddings + numpy cosine similarity.

    Per-label storage layout:
      base_dir/{label}.json   — chunk metadata (text, created_at, source)
      base_dir/{label}.npy    — embedding matrix (N, 384) float32
      base_dir/images/{label}/{timestamp}.jpg
    """

    def __init__(
        self,
        base_dir: str,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        clip_model_id: str = "clip-ViT-B-32",
        text_device: str = "cpu",
        clip_device: str = "cpu",
    ):
        from sentence_transformers import SentenceTransformer
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[RagStore] Loading text embedding model {model_id} on {text_device}...")
        self._embedder = SentenceTransformer(model_id, device=text_device)
        print(f"[RagStore] Loading CLIP model {clip_model_id} on {clip_device}...")
        self._clip_embedder = SentenceTransformer(clip_model_id, device=clip_device)

    def _get_embedder(self):
        return self._embedder

    def _get_clip_embedder(self):
        return self._clip_embedder

    # ── path helpers ──────────────────────────────────────────────────────────

    def _safe(self, label: str) -> str:
        return re.sub(r"[^\w\-:.]+", "_", label.strip()) or "default"

    def _meta_path(self, label: str) -> Path:
        return self.base_dir / f"{self._safe(label)}.json"

    def _emb_path(self, label: str) -> Path:
        return self.base_dir / f"{self._safe(label)}.npy"

    def _img_dir(self, label: str) -> Path:
        d = self.base_dir / "images" / self._safe(label)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _clip_emb_path(self, label: str) -> Path:
        return self.base_dir / f"{self._safe(label)}_clip.npy"

    # ── chunk helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, size: int = CHUNK_SIZE_WORDS) -> List[str]:
        words = text.split()
        return [
            " ".join(words[i : i + size])
            for i in range(0, len(words), size)
            if words[i : i + size]
        ]

    # ── metadata load/save ────────────────────────────────────────────────────

    def _load_meta(self, label: str) -> dict:
        path = self._meta_path(label)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"label": label, "chunks": []}

    def _save_meta(self, label: str, meta: dict) -> None:
        with open(self._meta_path(label), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _load_embeddings(self, label: str) -> Optional[np.ndarray]:
        path = self._emb_path(label)
        return np.load(str(path)) if path.exists() else None

    def _save_embeddings(self, label: str, embs: np.ndarray) -> None:
        np.save(str(self._emb_path(label)), embs)

    # ── public API ────────────────────────────────────────────────────────────

    def add_text(self, label: str, text: str, source: str = "ocr") -> None:
        chunks = self._chunk_text(text)
        if not chunks:
            return
        model = self._get_embedder()
        new_embs = model.encode(chunks, show_progress_bar=False).astype(np.float32)

        meta = self._load_meta(label)
        now = datetime.now(timezone.utc).isoformat()
        for chunk in chunks:
            meta["chunks"].append({"text": chunk, "created_at": now, "source": source})
        self._save_meta(label, meta)

        existing = self._load_embeddings(label)
        combined = np.vstack([existing, new_embs]) if existing is not None else new_embs
        self._save_embeddings(label, combined)

    def add_image(self, label: str, image: np.ndarray, description: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        img_path = self._img_dir(label) / f"{ts}.jpg"
        cv2.imwrite(str(img_path), image)
        self.add_text(label, description, source=f"image:{img_path.name}")

    def add_object(self, label: str, image: np.ndarray, description: str) -> None:
        """Store a named object: text embedding of description (384-dim) + CLIP image embedding (512-dim)."""
        # Text embedding of description for text-query retrieval
        self.add_text(label, description, source="object_description")

        # CLIP image embedding for future visual search
        try:
            from PIL import Image as PILImage
            clip_model = self._get_clip_embedder()
            pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_emb = clip_model.encode([pil_img], show_progress_bar=False).astype(np.float32)

            clip_path = self._clip_emb_path(label)
            existing = np.load(str(clip_path)) if clip_path.exists() else None
            combined = np.vstack([existing, img_emb]) if existing is not None else img_emb
            np.save(str(clip_path), combined)
        except Exception as e:
            print(f"[RagStore] CLIP image embedding failed: {e}")

        # Save image to disk
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        img_path = self._img_dir(label) / f"{ts}.jpg"
        cv2.imwrite(str(img_path), image)

    def query_global(self, question: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search across ALL labels. Returns [(chunk_text, label, score)] sorted by score desc."""
        model = self._get_embedder()
        q_emb = model.encode([question], show_progress_bar=False).astype(np.float32)[0]
        q_n = q_emb / max(float(np.linalg.norm(q_emb)), 1e-9)

        all_results: List[Tuple[str, str, float]] = []

        for meta_path in self.base_dir.glob("*.json"):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            label = meta.get("label", meta_path.stem)
            chunks = meta.get("chunks", [])
            if not chunks:
                continue

            emb_path = meta_path.with_suffix(".npy")
            if not emb_path.exists():
                continue

            try:
                embs = np.load(str(emb_path))
            except Exception:
                continue

            if embs.shape[0] != len(chunks):
                continue

            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs_n = embs / np.maximum(norms, 1e-9)
            scores = embs_n @ q_n

            for i, score in enumerate(scores):
                all_results.append((chunks[i]["text"], label, float(score)))

        all_results.sort(key=lambda x: x[2], reverse=True)
        return all_results[:top_k]

    def query(self, label: str, question: str, top_k: int = 3) -> List[str]:
        meta = self._load_meta(label)
        chunks = meta.get("chunks", [])
        if not chunks:
            return []
        embs = self._load_embeddings(label)
        if embs is None:
            return []

        model = self._get_embedder()
        q_emb = model.encode([question], show_progress_bar=False).astype(np.float32)[0]

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_n = embs / np.maximum(norms, 1e-9)
        q_n = q_emb / max(float(np.linalg.norm(q_emb)), 1e-9)
        scores = embs_n @ q_n

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [chunks[i]["text"] for i in top_idx if i < len(chunks)]

    def get_full_text(self, label: str) -> str:
        meta = self._load_meta(label)
        return "\n".join(c["text"] for c in meta.get("chunks", []))
