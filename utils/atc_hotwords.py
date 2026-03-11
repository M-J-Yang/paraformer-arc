from __future__ import annotations

import json
from pathlib import Path

_HOTWORD_CACHE: dict[tuple[str, str, int, int, float, float], list[str]] = {}


DEFAULT_EXTRA_HOTWORDS = (
    "跑道",
    "着陆",
    "高度",
    "起飞",
    "转弯",
    "检查",
    "请求",
    "场压",
    "保持",
    "航向",
    "左转",
    "右转",
    "进近",
    "复飞",
    "地面",
    "塔台",
    "幺拐",
    "摇拐",
    "洞拐",
    "零拐",
    "幺两",
    "幺二",
    "洞三",
    "零三",
    "07",
    "17",
    "03",
    "12",
)


def load_hotwords(
    wordlist_path: Path | None,
    vocab_freq_path: Path | None,
    max_hotwords: int = 300,
    min_freq: int = 2,
) -> list[str]:
    if wordlist_path is None and vocab_freq_path is None:
        return []

    cache_key = (
        str(wordlist_path or ""),
        str(vocab_freq_path or ""),
        max_hotwords,
        min_freq,
        _mtime_or_zero(wordlist_path),
        _mtime_or_zero(vocab_freq_path),
    )
    cached = _HOTWORD_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    hotwords: list[str] = []
    seen: set[str] = set()

    def add_token(token: str) -> None:
        normalized = token.strip()
        if not normalized or normalized in seen:
            return
        if _looks_like_noise(normalized):
            return
        seen.add(normalized)
        hotwords.append(normalized)

    for token in DEFAULT_EXTRA_HOTWORDS:
        add_token(token)

    if vocab_freq_path is not None and vocab_freq_path.exists():
        try:
            vocab_freq = json.loads(vocab_freq_path.read_text(encoding="utf-8"))
        except Exception:
            vocab_freq = {}
        if isinstance(vocab_freq, dict):
            for token, freq in sorted(vocab_freq.items(), key=lambda item: int(item[1]), reverse=True):
                if len(hotwords) >= max_hotwords:
                    break
                try:
                    if int(freq) < min_freq:
                        continue
                except Exception:
                    continue
                add_token(str(token))

    if wordlist_path is not None and wordlist_path.exists() and len(hotwords) < max_hotwords:
        try:
            tokens = wordlist_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            tokens = []
        for token in tokens:
            if len(hotwords) >= max_hotwords:
                break
            add_token(token)

    result = hotwords[:max_hotwords]
    _HOTWORD_CACHE[cache_key] = list(result)
    return result


def build_hotword_prompt(
    wordlist_path: Path | None,
    vocab_freq_path: Path | None,
    max_hotwords: int = 300,
    min_freq: int = 2,
) -> str:
    hotwords = load_hotwords(
        wordlist_path=wordlist_path,
        vocab_freq_path=vocab_freq_path,
        max_hotwords=max_hotwords,
        min_freq=min_freq,
    )
    return ", ".join(hotwords)


def _looks_like_noise(token: str) -> bool:
    stripped = token.strip()
    if not stripped:
        return True
    if len(stripped) == 1 and not stripped.isdigit() and not stripped.isalpha():
        return True
    if all(not ch.isalnum() and not ("\u4e00" <= ch <= "\u9fff") for ch in stripped):
        return True
    return False


def _mtime_or_zero(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
