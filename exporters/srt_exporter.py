from __future__ import annotations

from pathlib import Path

from models.result_schema import RecognitionResult, RecognitionSegment
from utils.time_format import format_srt_timestamp


def export_srt(result: RecognitionResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments = result.segments or [
        RecognitionSegment(index=1, start_ms=0, end_ms=0, text=result.full_text.strip())
    ]

    with output_path.open("w", encoding="utf-8") as file:
        for segment in segments:
            if not segment.text.strip():
                continue
            file.write(f"{segment.index}\n")
            file.write(
                f"{format_srt_timestamp(segment.start_ms)} --> "
                f"{format_srt_timestamp(segment.end_ms)}\n"
            )
            file.write(f"{segment.text}\n\n")
    return output_path
