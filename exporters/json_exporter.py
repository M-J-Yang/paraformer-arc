from __future__ import annotations

import json
from pathlib import Path

from models.result_schema import RecognitionResult


def export_json(result: RecognitionResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path
