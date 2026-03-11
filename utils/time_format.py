from __future__ import annotations


def format_srt_timestamp(total_ms: int) -> str:
    clamped_ms = max(0, int(total_ms))
    hours, remainder = divmod(clamped_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
