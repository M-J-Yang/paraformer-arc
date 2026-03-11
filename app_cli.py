from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ATC Phase 1 command-line recognizer",
    )
    parser.add_argument("--audio", required=True, help="Path to a single audio file")
    parser.add_argument("--output-dir", default="", help="Directory for exported files")
    parser.add_argument("--asr-model", default="", help="Local ASR model directory")
    parser.add_argument("--vad-model", default="", help="Local VAD model directory")
    parser.add_argument("--punc-model", default="", help="Local punctuation model directory")
    parser.add_argument("--hotword-wordlist", default="", help="ATC hotword wordlist file")
    parser.add_argument("--hotword-vocab-freq", default="", help="ATC hotword frequency json file")
    parser.add_argument("--text-rules", default="", help="Custom ATC text mapping rules json file")
    parser.add_argument(
        "--device",
        default="gpu",
        help='Inference device: "gpu" or "cpu"',
    )
    parser.add_argument(
        "--export-formats",
        default="txt,srt,json",
        help='Comma-separated formats, for example: "txt,srt,json"',
    )
    parser.add_argument(
        "--max-single-segment-time",
        type=int,
        default=30000,
        help="Maximum VAD segment length in milliseconds",
    )
    parser.add_argument(
        "--batch-size-s",
        type=int,
        default=20,
        help="FunASR batch_size_s",
    )
    parser.add_argument(
        "--batch-size-threshold-s",
        type=int,
        default=10,
        help="FunASR batch_size_threshold_s",
    )
    parser.add_argument(
        "--keep-temp-segments",
        action="store_true",
        help="Keep temporary WAV segments for debugging",
    )
    parser.add_argument(
        "--temp-dir",
        default="",
        help="Directory for temporary VAD segments",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from engines.recognition_service import RecognitionService
    from exporters import export_result, parse_export_formats
    from utils.config import (
        build_default_output_dir,
        build_default_temp_dir,
        resolve_asr_model_dir,
        resolve_hotword_vocab_freq_path,
        resolve_hotword_wordlist_path,
        resolve_punc_model_dir,
        resolve_text_rules_path,
        resolve_vad_model_dir,
    )
    from utils.device import resolve_device

    audio_path = Path(args.audio).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else build_default_output_dir(audio_path)
    )
    temp_dir = (
        Path(args.temp_dir).expanduser().resolve()
        if args.temp_dir
        else build_default_temp_dir(audio_path)
    )

    device = resolve_device(args.device)
    export_formats = parse_export_formats(args.export_formats)
    asr_model_dir = resolve_asr_model_dir(args.asr_model)
    vad_model_dir = resolve_vad_model_dir(args.vad_model)
    punc_model_dir = resolve_punc_model_dir(args.punc_model)
    hotword_wordlist_path = resolve_hotword_wordlist_path(args.hotword_wordlist)
    hotword_vocab_freq_path = resolve_hotword_vocab_freq_path(args.hotword_vocab_freq)
    text_rules_path = resolve_text_rules_path(args.text_rules)

    service = RecognitionService(
        asr_model_dir=asr_model_dir,
        vad_model_dir=vad_model_dir,
        punc_model_dir=punc_model_dir,
        device=device,
        enable_asr=True,
        enable_vad=True,
        enable_punc=True,
        max_single_segment_time=args.max_single_segment_time,
        batch_size_s=args.batch_size_s,
        batch_size_threshold_s=args.batch_size_threshold_s,
        hotword_wordlist_path=hotword_wordlist_path,
        hotword_vocab_freq_path=hotword_vocab_freq_path,
        text_rules_path=text_rules_path,
    )

    result = service.recognize(
        audio_path=audio_path,
        temp_dir=temp_dir,
        keep_temp_segments=args.keep_temp_segments,
    )
    exported_files = export_result(
        result=result,
        output_dir=output_dir,
        export_formats=export_formats,
    )

    print(f"[OK] Audio: {result.audio.path}")
    print(f"[OK] Device: {device}")
    print(f"[OK] Segments: {len(result.segments)}")
    print(f"[OK] Output: {output_dir}")
    for exported_file in exported_files:
        print(f"[OK] Exported: {exported_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
