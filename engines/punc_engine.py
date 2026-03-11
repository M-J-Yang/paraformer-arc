from __future__ import annotations

from pathlib import Path

from funasr import AutoModel


class PunctuationEngine:
    def __init__(self, model_dir: Path, device: str) -> None:
        self.model = AutoModel(
            model=str(model_dir),
            device=device,
            disable_update=True,
            disable_pbar=True,
        )

    def punctuate(self, text: str) -> str:
        if not text.strip():
            return ""
        result = self.model.generate(input=text, disable_pbar=True)
        if not result:
            return text.strip()
        return (result[0].get("text") or text).strip()
