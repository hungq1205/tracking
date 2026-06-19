import re
from datetime import datetime, timezone
from pathlib import Path

from domain.types import MemoryDocument, MemoryEntry

MIN_OVERLAP_CHARS = 20
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def filter_new_sentences(block_text: str, existing: str, min_chars: int = 15) -> str:
    """Return the portion of block_text not already present in existing.

    Splits block_text into sentences and drops any whose normalized form appears
    as a substring of the normalized existing text. Returns the remaining sentences
    joined by spaces, or "" if everything was already seen.
    """
    block_text = block_text.strip()
    if not block_text:
        return ""
    if not existing:
        return block_text

    existing_norm = normalize_text(existing)
    raw_sentences = _SENT_SPLIT_RE.split(block_text)
    new_sentences = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_norm = normalize_text(sent)
        if len(sent_norm) >= min_chars and sent_norm in existing_norm:
            continue
        new_sentences.append(sent)
    return " ".join(new_sentences)


def find_overlap_suffix_prefix(stored: str, new: str, min_overlap: int = MIN_OVERLAP_CHARS) -> int:
    """Return length of longest suffix of stored matching prefix of new."""
    stored_norm = normalize_text(stored)
    new_norm = normalize_text(new)
    if not stored_norm or not new_norm:
        return 0
    max_len = min(len(stored_norm), len(new_norm))
    best = 0
    for size in range(max_len, min_overlap - 1, -1):
        if stored_norm[-size:] == new_norm[:size]:
            best = size
            break
    return best


class JsonMemoryStore:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, label: str) -> Path:
        safe = re.sub(r"[^\w\-:.]+", "_", label.strip()) or "default"
        return self.base_dir / f"{safe}.json"

    def load(self, label: str) -> MemoryDocument:
        path = self._path_for(label)
        if not path.exists():
            return MemoryDocument(label=label)
        import json

        with open(path, "r", encoding="utf-8") as f:
            return MemoryDocument.from_dict(json.load(f))

    def save(self, doc: MemoryDocument) -> None:
        import json

        path = self._path_for(doc.label)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)

    def append(self, label: str, text: str, source: str = "ocr") -> tuple[str, str]:
        """Append text with overlap dedup. Returns (appended_text, full_text)."""
        text = text.strip()
        if not text:
            doc = self.load(label)
            return "", doc.full_text

        doc = self.load(label)
        overlap = find_overlap_suffix_prefix(doc.full_text, text)
        if overlap > 0:
            new_norm = normalize_text(text)
            stored_norm = normalize_text(doc.full_text)
            # Map normalized overlap back to original new text by proportional trim
            ratio = overlap / max(len(new_norm), 1)
            trim_idx = int(len(text) * ratio)
            appended = text[trim_idx:].strip()
        else:
            appended = text

        if not appended:
            return "", doc.full_text

        now = datetime.now(timezone.utc).isoformat()
        doc.entries.append(MemoryEntry(text=appended, created_at=now, source=source))
        if doc.full_text:
            doc.full_text = f"{doc.full_text}\n{appended}"
        else:
            doc.full_text = appended
        self.save(doc)
        return appended, doc.full_text

    def get_full_text(self, label: str) -> str:
        return self.load(label).full_text
