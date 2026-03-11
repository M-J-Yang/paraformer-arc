from __future__ import annotations

from pathlib import Path

from funasr import AutoModel


class VADEngine:
    def __init__(self, model_dir: Path, device: str) -> None:
        self.model = AutoModel(
            model=str(model_dir),
            device=device,
            disable_update=True,
            disable_pbar=True,
        )

    def detect(self, audio_path: Path, max_single_segment_time: int) -> list[list[int]]:
        result = self.model.generate(
            input=str(audio_path),
            max_single_segment_time=max_single_segment_time,
            disable_pbar=True,
        )
        if not result:
            return []
        return result[0].get("value", []) or []
