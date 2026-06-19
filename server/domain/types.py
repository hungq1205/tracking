from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MemoryEntry:
    text: str
    created_at: str
    source: str = "ocr"


@dataclass
class MemoryDocument:
    label: str
    entries: List[MemoryEntry] = field(default_factory=list)
    full_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "entries": [
                {"text": e.text, "created_at": e.created_at, "source": e.source}
                for e in self.entries
            ],
            "full_text": self.full_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryDocument":
        entries = [
            MemoryEntry(
                text=e["text"],
                created_at=e["created_at"],
                source=e.get("source", "ocr"),
            )
            for e in data.get("entries", [])
        ]
        return cls(
            label=data.get("label", ""),
            entries=entries,
            full_text=data.get("full_text", ""),
        )


@dataclass
class ImageMemoryEntry:
    image_path: str
    description: str
    created_at: str
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "description": self.description,
            "created_at": self.created_at,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageMemoryEntry":
        return cls(
            image_path=data.get("image_path", ""),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            label=data.get("label", ""),
        )
