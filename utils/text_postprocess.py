from __future__ import annotations

import json
import re
from pathlib import Path

_TEXT_RULES_CACHE: dict[tuple[str, float], dict[str, object]] = {}

# Spoken ATC numerals frequently appear in heading/runway/altitude phrases.
ATC_DIGIT_MAP = {
    "洞": "0",
    "零": "0",
    "幺": "1",
    "一": "1",
    "两": "2",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "拐": "7",
    "七": "7",
    "八": "8",
    "九": "9",
}
ATC_SPECIAL_DIGIT_CHARS = {"洞", "零", "幺", "两", "拐"}
ATC_NUMERIC_KEYWORDS = (
    "跑道",
    "航向",
    "高度",
    "高",
    "下到",
    "上升到",
    "下降到",
    "保持",
    "地面风",
    "修正海压",
)


def apply_text_postprocess(text: str, rules_path: Path | None) -> str:
    normalized = text.strip()
    if not normalized:
        return normalized

    normalized = normalize_atc_numeric_phrases(normalized)

    rules = load_text_rules(rules_path)
    replace_rules = rules.get("replace", rules.get("direct_mappings", {}))
    if isinstance(replace_rules, dict):
        for source, target in sorted(replace_rules.items(), key=lambda item: len(str(item[0])), reverse=True):
            normalized = normalized.replace(str(source), str(target))

    return normalized.strip()


def normalize_atc_numeric_phrases(text: str) -> str:
    keywords = "|".join(re.escape(keyword) for keyword in ATC_NUMERIC_KEYWORDS)
    spoken_digits = "".join(ATC_DIGIT_MAP.keys())

    keyword_pattern = re.compile(rf"({keywords})([{spoken_digits}]{{2,4}})")
    normalized = keyword_pattern.sub(
        lambda match: f"{match.group(1)}{_convert_digit_phrase(match.group(2), require_special=False)}",
        text,
    )

    standalone_pattern = re.compile(rf"(?<!\d)([{spoken_digits}]{{2,4}})(?!\d)")
    normalized = standalone_pattern.sub(
        lambda match: _convert_digit_phrase(match.group(1), require_special=True),
        normalized,
    )
    return normalized


def _convert_digit_phrase(phrase: str, require_special: bool) -> str:
    if not phrase or any(ch not in ATC_DIGIT_MAP for ch in phrase):
        return phrase
    if require_special and not any(ch in ATC_SPECIAL_DIGIT_CHARS for ch in phrase):
        return phrase
    return "".join(ATC_DIGIT_MAP[ch] for ch in phrase)


def load_text_rules(rules_path: Path | None) -> dict[str, object]:
    if rules_path is None or not rules_path.exists():
        return {}

    cache_key = (str(rules_path), _mtime_or_zero(rules_path))
    cached = _TEXT_RULES_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    try:
        data = json.loads(rules_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    result = data if isinstance(data, dict) else {}
    _TEXT_RULES_CACHE[cache_key] = dict(result)
    return result


def _mtime_or_zero(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
