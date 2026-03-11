from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from models.app_settings import AppSettings

SOURCE_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else SOURCE_ROOT
PROJECT_ROOT = APP_ROOT
SEARCH_ROOTS = list(dict.fromkeys([APP_ROOT, APP_ROOT.parent, Path.cwd().resolve()]))
SETTINGS_FILE = APP_ROOT / ".atc_gui_settings.json"


def default_output_root_dir() -> Path:
    return APP_ROOT / "inference" / "infer_outputs"


def normalize_output_root_dir(raw_path: str) -> Path:
    return Path(raw_path).expanduser().resolve() if raw_path.strip() else default_output_root_dir().resolve()


def build_default_output_dir(audio_path: Path) -> Path:
    return default_output_root_dir() / audio_path.stem


def build_default_temp_dir(audio_path: Path) -> Path:
    return APP_ROOT / "inference" / "_tmp_segments" / audio_path.stem


def build_warmup_temp_dir() -> Path:
    return APP_ROOT / "inference" / "_tmp_warmup"


def build_export_output_dir(base_output_dir: Path, audio_stem: str) -> Path:
    return base_output_dir / audio_stem


def resolve_hotword_wordlist_path(explicit_path: str = "") -> Path | None:
    return _resolve_optional_path(
        explicit_path=explicit_path,
        env_names=("ATC_HOTWORD_WORDLIST",),
        candidates=[
            root / "paraformer-arc" / "chinese_ATC_formatted" / "TXTdata" / "wordlist.txt"
            for root in SEARCH_ROOTS
        ],
    )


def resolve_hotword_vocab_freq_path(explicit_path: str = "") -> Path | None:
    return _resolve_optional_path(
        explicit_path=explicit_path,
        env_names=("ATC_HOTWORD_VOCAB_FREQ",),
        candidates=[
            root / "paraformer-arc" / "chinese_ATC_formatted" / "TXTdata" / "extracted_vocab_freq.json"
            for root in SEARCH_ROOTS
        ],
    )


def resolve_text_rules_path(explicit_path: str = "") -> Path | None:
    return _resolve_optional_path(
        explicit_path=explicit_path,
        env_names=("ATC_TEXT_RULES",),
        candidates=[
            root / "config" / "atc_text_rules.json"
            for root in SEARCH_ROOTS
        ],
    )


def load_app_settings() -> AppSettings:
    if not SETTINGS_FILE.exists():
        return AppSettings(output_dir=str(default_output_root_dir()))
    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return AppSettings(output_dir=str(default_output_root_dir()))
    settings = AppSettings.from_dict(data)
    if not settings.output_dir:
        settings.output_dir = str(default_output_root_dir())
    return settings


def save_app_settings(settings: AppSettings) -> Path:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump(settings.to_dict(), file, ensure_ascii=False, indent=2)
        file.write("\n")
    return SETTINGS_FILE


def resolve_asr_model_dir(explicit_path: str) -> Path:
    candidates = _build_model_candidates(
        "paraformer_v1/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "asr",
        "models/paraformer_v1/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    )
    return _resolve_model_dir(
        explicit_path=explicit_path,
        env_names=("ASR_MODEL_DIR", "MODEL_DIR"),
        candidates=candidates,
        label="ASR",
        required_files=("config.yaml", "model.pt", "tokens.json", "seg_dict", "am.mvn"),
    )


def resolve_vad_model_dir(explicit_path: str) -> Path:
    candidates = _build_model_candidates(
        "speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "vad",
        "models/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    )
    return _resolve_model_dir(
        explicit_path=explicit_path,
        env_names=("VAD_MODEL_DIR", "VAD_MODEL"),
        candidates=candidates,
        label="VAD",
        required_files=("config.yaml", "model.pt"),
    )


def resolve_punc_model_dir(explicit_path: str) -> Path:
    candidates = _build_model_candidates(
        "punc_ct-transformer_cn-en-common-vocab471067-large",
        "punc",
        "models/punc_ct-transformer_cn-en-common-vocab471067-large",
    )
    return _resolve_model_dir(
        explicit_path=explicit_path,
        env_names=("PUNC_MODEL_DIR", "PUNC_MODEL"),
        candidates=candidates,
        label="PUNC",
        required_files=("config.yaml", "model.pt"),
    )


def _resolve_model_dir(
    explicit_path: str,
    env_names: tuple[str, ...],
    candidates: list[Path],
    label: str,
    required_files: tuple[str, ...],
) -> Path:
    raw_candidates: list[Path] = []
    if explicit_path:
        raw_candidates.append(Path(explicit_path).expanduser())

    for env_name in env_names:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            raw_candidates.append(Path(env_value).expanduser())

    raw_candidates.extend(candidates)

    checked_candidates: list[Path] = []
    for candidate in raw_candidates:
        path = candidate.resolve()
        if path in checked_candidates:
            continue
        checked_candidates.append(path)
        if _contains_required_files(path, required_files):
            return path

    searched_paths = "\n".join(f"  - {path}" for path in checked_candidates)
    raise FileNotFoundError(
        f"{label} model directory was not found. Checked:\n{searched_paths}"
    )


def _build_model_candidates(*relative_paths: str) -> list[Path]:
    candidates: list[Path] = []
    for root in SEARCH_ROOTS:
        for relative_path in relative_paths:
            path = Path(relative_path)
            if path.parts and path.parts[0] == "models":
                candidates.append(root / path)
            else:
                candidates.append(root / "model_store" / path)
    return candidates


def _resolve_optional_path(
    explicit_path: str,
    env_names: tuple[str, ...],
    candidates: list[Path],
) -> Path | None:
    raw_candidates: list[Path] = []
    if explicit_path:
        raw_candidates.append(Path(explicit_path).expanduser())

    for env_name in env_names:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            raw_candidates.append(Path(env_value).expanduser())

    raw_candidates.extend(candidates)

    checked_candidates: list[Path] = []
    for candidate in raw_candidates:
        path = candidate.resolve()
        if path in checked_candidates:
            continue
        checked_candidates.append(path)
        if path.exists():
            return path
    return None


def _contains_required_files(path: Path, required_files: tuple[str, ...]) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return all((path / required_file).exists() for required_file in required_files)
