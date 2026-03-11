from __future__ import annotations

from pathlib import Path

from exporters.json_exporter import export_json
from exporters.srt_exporter import export_srt
from exporters.txt_exporter import export_txt
from models.result_schema import RecognitionResult

SUPPORTED_EXPORT_FORMATS = ("txt", "srt", "json")


def parse_export_formats(value: str) -> list[str]:
    formats = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not formats:
        raise ValueError("At least one export format is required")

    invalid_formats = [item for item in formats if item not in SUPPORTED_EXPORT_FORMATS]
    if invalid_formats:
        supported = ", ".join(SUPPORTED_EXPORT_FORMATS)
        raise ValueError(f"Unsupported export formats: {', '.join(invalid_formats)}. Supported: {supported}")

    deduplicated_formats: list[str] = []
    for export_format in formats:
        if export_format not in deduplicated_formats:
            deduplicated_formats.append(export_format)
    return deduplicated_formats


def export_result(
    result: RecognitionResult,
    output_dir: Path,
    export_formats: list[str],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_files: list[Path] = []

    for export_format in export_formats:
        if export_format == "txt":
            exported_files.append(export_txt(result, output_dir / "result.txt"))
        elif export_format == "srt":
            exported_files.append(export_srt(result, output_dir / "result.srt"))
        elif export_format == "json":
            exported_files.append(export_json(result, output_dir / "result.json"))

    return exported_files


__all__ = ["SUPPORTED_EXPORT_FORMATS", "export_result", "parse_export_formats"]
