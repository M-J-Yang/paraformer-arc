from __future__ import annotations

from pathlib import Path

from models.result_schema import AudioSource

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


def load_audio_source(audio_path: Path) -> AudioSource:
    path = audio_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Audio path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio format: {path.suffix}. Supported: {supported}")

    return AudioSource(
        path=str(path),
        name=path.name,
        stem=path.stem,
        suffix=path.suffix.lower(),
    )


def collect_audio_files(audio_dir: Path | str) -> list[Path]:
    directory = Path(audio_dir).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Audio folder not found: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Audio path is not a folder: {directory}")
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
