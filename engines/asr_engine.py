from __future__ import annotations

from pathlib import Path
from typing import Callable

from funasr import AutoModel


class ASREngine:
    def __init__(
        self,
        model_dir: Path,
        device: str,
        batch_size_s: int,
        batch_size_threshold_s: int,
        hotword_provider: Callable[[], str] | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.batch_size_s = batch_size_s
        self.batch_size_threshold_s = batch_size_threshold_s
        self.hotword_provider = hotword_provider
        self.model = AutoModel(
            model=str(model_dir),
            init_param=str(model_dir / "model.pt"),
            tokenizer_conf={
                "token_list": str(model_dir / "tokens.json"),
                "seg_dict_file": str(model_dir / "seg_dict"),
            },
            frontend_conf={"cmvn_file": str(model_dir / "am.mvn")},
            device=device,
            disable_update=True,
            disable_pbar=True,
        )

    def transcribe(self, audio_path: Path) -> str:
        generate_kwargs = {
            "input": str(audio_path),
            "batch_size_s": self.batch_size_s,
            "batch_size_threshold_s": self.batch_size_threshold_s,
            "disable_pbar": True,
        }
        hotword = self.hotword_provider() if self.hotword_provider is not None else ""
        if hotword:
            generate_kwargs["hotword"] = hotword
        result = self.model.generate(**generate_kwargs)
        if not result:
            return ""
        return (result[0].get("text") or "").strip()
