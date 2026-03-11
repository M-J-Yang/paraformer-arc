from __future__ import annotations

from pathlib import Path

from models.result_schema import RecognitionResult


def export_txt(result: RecognitionResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = result.full_text.strip()
    if not text and result.segments:
        text = "\n".join(segment.text for segment in result.segments if segment.text.strip())

    with output_path.open("w", encoding="utf-8") as file:
        file.write(text)
        file.write("\n")
    return output_path
