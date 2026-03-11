from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from engines.asr_engine import ASREngine
from engines.audio_loader import load_audio_source
from engines.punc_engine import PunctuationEngine
from engines.vad_engine import VADEngine
from models.result_schema import RecognitionResult, RecognitionSegment
from utils.atc_hotwords import build_hotword_prompt, load_hotwords
from utils.config import build_warmup_temp_dir
from utils.text_postprocess import apply_text_postprocess


class RecognitionService:
    def __init__(
        self,
        asr_model_dir: Path | None,
        vad_model_dir: Path | None,
        punc_model_dir: Path | None,
        device: str,
        enable_asr: bool,
        enable_vad: bool,
        enable_punc: bool,
        max_single_segment_time: int,
        batch_size_s: int,
        batch_size_threshold_s: int,
        hotword_wordlist_path: Path | None = None,
        hotword_vocab_freq_path: Path | None = None,
        text_rules_path: Path | None = None,
    ) -> None:
        self.device = device
        self.enable_asr = enable_asr
        self.enable_vad = enable_vad
        self.enable_punc = enable_punc
        self.max_single_segment_time = max_single_segment_time
        self.hotword_wordlist_path = hotword_wordlist_path
        self.hotword_vocab_freq_path = hotword_vocab_freq_path
        self.text_rules_path = text_rules_path
        self.asr_engine = (
            ASREngine(
                model_dir=asr_model_dir,
                device=device,
                batch_size_s=batch_size_s,
                batch_size_threshold_s=batch_size_threshold_s,
                hotword_provider=self._get_hotword_prompt,
            )
            if enable_asr and asr_model_dir is not None
            else None
        )
        self.vad_engine = VADEngine(model_dir=vad_model_dir, device=device) if enable_vad and vad_model_dir is not None else None
        self.punc_engine = (
            PunctuationEngine(model_dir=punc_model_dir, device=device)
            if enable_punc and punc_model_dir is not None
            else None
        )
        self.model_paths = {
            "asr": str(asr_model_dir) if asr_model_dir is not None else "",
            "vad": str(vad_model_dir) if vad_model_dir is not None else "",
            "punc": str(punc_model_dir) if punc_model_dir is not None else "",
        }

    def warmup(self, on_log: Callable[[str], None] | None = None) -> None:
        warmup_dir = build_warmup_temp_dir()
        warmup_audio = warmup_dir / "warmup.wav"
        warmup_dir.mkdir(parents=True, exist_ok=True)
        self._write_silence_wav(warmup_audio, duration_ms=1000)
        try:
            if self.vad_engine is not None:
                self._emit_log(on_log, "[INFO] 预热 VAD")
                self.vad_engine.detect(warmup_audio, max_single_segment_time=self.max_single_segment_time)
            if self.asr_engine is not None:
                self._emit_log(on_log, "[INFO] 预热 ASR")
                self.asr_engine.transcribe(warmup_audio)
            if self.punc_engine is not None:
                self._emit_log(on_log, "[INFO] 预热 PUNC")
                self.punc_engine.punctuate("模型预热")
        finally:
            try:
                if warmup_audio.exists():
                    warmup_audio.unlink()
            except OSError:
                pass

    def recognize(
        self,
        audio_path: Path,
        temp_dir: Path | None = None,
        keep_temp_segments: bool = False,
        on_status: Callable[[str], None] | None = None,
        on_log: Callable[[str], None] | None = None,
        on_progress: Callable[[int, str], None] | None = None,
    ) -> RecognitionResult:
        self._emit_status(on_status, "读取音频")
        audio = load_audio_source(audio_path)
        self._emit_log(on_log, f"[INFO] 音频已加载: {audio.path}")
        if self.asr_engine is None:
            raise RuntimeError("ASR 引擎未初始化，无法执行识别。")

        source_audio = Path(audio.path)
        if self.vad_engine is not None:
            self._emit_status(on_status, "执行 VAD 分段")
            self._emit_progress(on_progress, 5, "执行 VAD 分段")
            vad_segments = self.vad_engine.detect(
                source_audio,
                max_single_segment_time=self.max_single_segment_time,
            )
            self._emit_log(on_log, f"[INFO] VAD 分段数: {len(vad_segments)}")
        else:
            duration_ms = self._probe_audio_duration_ms(source_audio)
            vad_segments = [[0, duration_ms]]
            self._emit_log(on_log, "[INFO] VAD 已关闭，整段音频直接送入识别")

        segments: list[RecognitionSegment] = []
        raw_segments: list[dict[str, object]] = []
        total_segments = max(len(vad_segments), 1)

        if self.vad_engine is None:
            full_text = self.asr_engine.transcribe(source_audio)
            final_text = self._postprocess_text(self._apply_punctuation(full_text))
            self._emit_log(on_log, f"[INFO] 整段识别结果: {final_text[:40] if final_text else '<empty>'}")
            raw_segments.append(
                {
                    "segment_index": 1,
                    "start_ms": 0,
                    "end_ms": int(vad_segments[0][1]),
                    "raw_text": full_text,
                    "text": final_text,
                }
            )
            if final_text:
                segments.append(
                    RecognitionSegment(
                        index=1,
                        start_ms=0,
                        end_ms=int(vad_segments[0][1]),
                        text=final_text,
                    )
                )
            self._emit_status(on_status, "识别完成")
            self._emit_progress(on_progress, 100, "识别完成")
            return RecognitionResult(
                audio=audio,
                device=self.device,
                created_at=datetime.now(timezone.utc).isoformat(),
                full_text="\n".join(segment.text for segment in segments),
                segments=segments,
                model_paths=self.model_paths,
                raw_result={
                    "vad_enabled": False,
                    "punc_enabled": self.punc_engine is not None,
                    "vad_segments": vad_segments,
                    "segments": raw_segments,
                    "hotword_enabled": bool(self._get_hotword_prompt()),
                    "hotword_count": len(self._get_hotwords()),
                    "text_rules_path": str(self.text_rules_path) if self.text_rules_path is not None else "",
                },
            )

        with self._segment_workspace(temp_dir=temp_dir, keep=keep_temp_segments) as workspace:
            self._emit_log(on_log, f"[INFO] 临时分段目录: {workspace}")
            for index, vad_segment in enumerate(vad_segments, start=1):
                if not isinstance(vad_segment, (list, tuple)) or len(vad_segment) != 2:
                    continue

                start_ms = int(vad_segment[0])
                end_ms = int(vad_segment[1])
                if end_ms <= start_ms:
                    continue

                self._emit_status(on_status, f"识别分段 {index}/{total_segments}")
                segment_percent = 10 + int((index - 1) / total_segments * 80)
                self._emit_progress(on_progress, segment_percent, f"切分音频 {index}/{total_segments}")
                segment_audio_path = workspace / f"seg_{index:04d}.wav"
                self._extract_audio_segment(
                    source_audio=source_audio,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    output_audio=segment_audio_path,
                )

                self._emit_progress(on_progress, min(segment_percent + 5, 95), f"执行 ASR {index}/{total_segments}")
                text = self.asr_engine.transcribe(segment_audio_path)
                punctuated_text = self._apply_punctuation(text)
                final_text = self._postprocess_text(punctuated_text)
                preview = final_text[:40] if final_text else "<empty>"
                self._emit_log(on_log, f"[INFO] 分段 {index}: {preview}")

                raw_segments.append(
                    {
                        "segment_index": index,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "raw_text": text,
                        "punctuated_text": punctuated_text,
                        "text": final_text,
                    }
                )
                if not final_text:
                    continue

                segments.append(
                    RecognitionSegment(
                        index=index,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        text=final_text,
                    )
                )

        self._emit_status(on_status, "识别完成")
        self._emit_progress(on_progress, 100, "识别完成")
        return RecognitionResult(
            audio=audio,
            device=self.device,
            created_at=datetime.now(timezone.utc).isoformat(),
            full_text="\n".join(segment.text for segment in segments),
            segments=segments,
            model_paths=self.model_paths,
            raw_result={
                "vad_enabled": self.vad_engine is not None,
                "punc_enabled": self.punc_engine is not None,
                "vad_segments": vad_segments,
                "segments": raw_segments,
                "hotword_enabled": bool(self._get_hotword_prompt()),
                "hotword_count": len(self._get_hotwords()),
                "text_rules_path": str(self.text_rules_path) if self.text_rules_path is not None else "",
            },
        )

    def _apply_punctuation(self, text: str) -> str:
        if self.punc_engine is None:
            return text.strip()
        return self.punc_engine.punctuate(text)

    def _postprocess_text(self, text: str) -> str:
        return apply_text_postprocess(text, self.text_rules_path)

    def _get_hotwords(self) -> list[str]:
        return load_hotwords(
            wordlist_path=self.hotword_wordlist_path,
            vocab_freq_path=self.hotword_vocab_freq_path,
        )

    def _get_hotword_prompt(self) -> str:
        return build_hotword_prompt(
            wordlist_path=self.hotword_wordlist_path,
            vocab_freq_path=self.hotword_vocab_freq_path,
        )

    def _probe_audio_duration_ms(self, audio_path: Path) -> int:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError:
            return self.max_single_segment_time

        if completed.returncode != 0:
            return self.max_single_segment_time

        try:
            seconds = float((completed.stdout or "").strip())
        except ValueError:
            return self.max_single_segment_time
        return max(int(seconds * 1000), 1)

    @contextmanager
    def _segment_workspace(self, temp_dir: Path | None, keep: bool) -> Iterator[Path]:
        if temp_dir is None:
            with tempfile.TemporaryDirectory(prefix="atc_segments_") as tmp_dir:
                yield Path(tmp_dir)
            return

        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            yield temp_dir
        finally:
            if not keep:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _extract_audio_segment(
        self,
        source_audio: Path,
        start_ms: int,
        end_ms: int,
        output_audio: Path,
    ) -> None:
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        start_seconds = max(0.0, start_ms / 1000.0)
        end_seconds = max(start_seconds, end_ms / 1000.0)
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_seconds:.3f}",
            "-to",
            f"{end_seconds:.3f}",
            "-i",
            str(source_audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output_audio),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg was not found in PATH") from exc

        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "unknown ffmpeg error"
            raise RuntimeError(f"ffmpeg failed while extracting audio segment: {message}")

    @staticmethod
    def _write_silence_wav(audio_path: Path, duration_ms: int, sample_rate: int = 16000) -> None:
        frame_count = max(1, sample_rate * duration_ms // 1000)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * frame_count)

    @staticmethod
    def _emit_status(callback: Callable[[str], None] | None, message: str) -> None:
        if callback is not None:
            callback(message)

    @staticmethod
    def _emit_log(callback: Callable[[str], None] | None, message: str) -> None:
        if callback is not None:
            callback(message)

    @staticmethod
    def _emit_progress(
        callback: Callable[[int, str], None] | None,
        percent: int,
        message: str,
    ) -> None:
        if callback is not None:
            callback(max(0, min(percent, 100)), message)
