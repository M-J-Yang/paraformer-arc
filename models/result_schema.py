from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AudioSource:
    path: str
    name: str
    stem: str
    suffix: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "name": self.name,
            "stem": self.stem,
            "suffix": self.suffix,
        }


@dataclass(slots=True)
class RecognitionSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
        }


@dataclass(slots=True)
class RecognitionResult:
    audio: AudioSource
    device: str
    created_at: str
    full_text: str
    segments: list[RecognitionSegment]
    model_paths: dict[str, str]
    raw_result: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "audio": self.audio.to_dict(),
            "device": self.device,
            "created_at": self.created_at,
            "text": self.full_text,
            "segment_count": len(self.segments),
            "segments": [segment.to_dict() for segment in self.segments],
            "model_paths": self.model_paths,
            "raw_result": self.raw_result,
        }
