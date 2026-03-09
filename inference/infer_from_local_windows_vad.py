import argparse
import json
import subprocess
import traceback
from pathlib import Path

import torch
from funasr import AutoModel


def choose_device(force_device: str) -> str:
    if force_device:
        return force_device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def collect_audio_files(audio_dir: Path):
    exts = {".wav", ".mp3", ".flac", ".m4a"}
    return sorted([p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def to_ms(value):
    if value is None:
        return 0
    try:
        return int(round(float(value)))
    except Exception:
        return 0


def normalize_segments(item: dict):
    segments = []
    sentence_info = item.get("sentence_info") or []

    for sent in sentence_info:
        text = sent.get("text") or sent.get("sentence") or ""
        if "start_time" in sent and "end_time" in sent:
            start_ms = int(round(float(sent["start_time"]) * 1000.0))
            end_ms = int(round(float(sent["end_time"]) * 1000.0))
        else:
            start_ms = to_ms(sent.get("start", 0))
            end_ms = to_ms(sent.get("end", 0))
        segments.append({"start_ms": start_ms, "end_ms": end_ms, "text": text.strip()})

    if not segments:
        text = (item.get("text") or "").strip()
        ts = item.get("timestamp") or []
        if text and ts:
            start_ms = to_ms(ts[0][0])
            end_ms = to_ms(ts[-1][1])
            segments.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
        elif text:
            segments.append({"start_ms": 0, "end_ms": 0, "text": text})

    return segments


def write_outputs(out_dir: Path, result_item: dict, segments: list):
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result_item, f, ensure_ascii=False, indent=2)

    with (out_dir / "segments.txt").open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg['start_ms'] / 1000.0:.3f}\t{seg['end_ms'] / 1000.0:.3f}\t{seg['text']}\n")

    merged_text = "".join([seg["text"] for seg in segments]).strip()
    with (out_dir / "text.txt").open("w", encoding="utf-8") as f:
        f.write(merged_text + "\n")


def run_cmd(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def extract_segment_wav(src_audio: Path, start_ms: int, end_ms: int, dst_wav: Path):
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    start_s = max(0.0, start_ms / 1000.0)
    end_s = max(start_s, end_ms / 1000.0)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-i",
        str(src_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst_wav),
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(description="FunASR local inference with VAD and timestamps on Windows")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--audio-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--force-device", type=str, default="")
    parser.add_argument("--vad-model", type=str, default="")
    parser.add_argument("--punc-model", type=str, default="")
    parser.add_argument("--max-single-segment-time", type=int, default=30000)
    parser.add_argument("--batch-size-s", type=int, default=20)
    parser.add_argument("--batch-size-threshold-s", type=int, default=10)
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    model_dir = Path(args.model_dir) if args.model_dir else root_dir / "models" / "paraformer_v1" / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    audio_dir = Path(args.audio_dir) if args.audio_dir else root_dir / "audios"
    output_dir = Path(args.output_dir) if args.output_dir else root_dir / "infer_outputs"
    vad_model_dir = Path(args.vad_model) if args.vad_model else None
    punc_model_dir = Path(args.punc_model) if args.punc_model else None

    if not (model_dir / "config.yaml").exists():
        raise FileNotFoundError(f"Model config not found: {model_dir / 'config.yaml'}")
    if not (model_dir / "model.pt").exists():
        raise FileNotFoundError(f"Model weights not found: {model_dir / 'model.pt'}")
    if vad_model_dir is None:
        raise ValueError("Missing --vad-model path. Please pass a local VAD model directory.")
    if punc_model_dir is None:
        raise ValueError("Missing --punc-model path. Please pass a local PUNC model directory.")
    if not (vad_model_dir / "config.yaml").exists():
        raise FileNotFoundError(f"VAD model config not found: {vad_model_dir / 'config.yaml'}")
    if not (vad_model_dir / "model.pt").exists():
        raise FileNotFoundError(f"VAD model weights not found: {vad_model_dir / 'model.pt'}")
    if not (punc_model_dir / "config.yaml").exists():
        raise FileNotFoundError(f"PUNC model config not found: {punc_model_dir / 'config.yaml'}")
    if not (punc_model_dir / "model.pt").exists():
        raise FileNotFoundError(f"PUNC model weights not found: {punc_model_dir / 'model.pt'}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio folder not found: {audio_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    map_file = output_dir / "file_map.txt"

    device = choose_device(args.force_device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] ASR model: {model_dir}")
    print(f"[INFO] VAD model: {vad_model_dir}")
    print(f"[INFO] PUNC model: {punc_model_dir}")

    asr_model = AutoModel(
        model=str(model_dir),
        init_param=str(model_dir / "model.pt"),
        tokenizer_conf={
            "token_list": str(model_dir / "tokens.json"),
            "seg_dict_file": str(model_dir / "seg_dict"),
        },
        frontend_conf={"cmvn_file": str(model_dir / "am.mvn")},
        device=device,
        disable_update=True,
        disable_pbar=True,
    )
    vad_model = AutoModel(
        model=str(vad_model_dir),
        device=device,
        disable_update=True,
        disable_pbar=True,
    )
    punc_model = AutoModel(
        model=str(punc_model_dir),
        device=device,
        disable_update=True,
        disable_pbar=True,
    )

    audio_files = collect_audio_files(audio_dir)
    if not audio_files:
        print(f"[WARN] No audio files found in {audio_dir}")
        return 2

    failed = 0
    with map_file.open("w", encoding="utf-8") as mf:
        mf.write("# infer id mapping\n")

        for idx, audio_path in enumerate(audio_files, start=1):
            sample_id = f"sample_{idx:03d}"
            sample_out = output_dir / sample_id
            print(f"\n[INFO] Inference: {audio_path.name} (id={sample_id})")

            try:
                sample_out.mkdir(parents=True, exist_ok=True)
                tmp_dir = sample_out / "_tmp_segments"
                tmp_dir.mkdir(parents=True, exist_ok=True)

                vad_res = vad_model.generate(
                    input=str(audio_path),
                    max_single_segment_time=args.max_single_segment_time,
                    disable_pbar=True,
                )
                if not vad_res:
                    raise RuntimeError("Empty VAD result")

                vad_segments = vad_res[0].get("value", [])
                segments = []
                per_seg_results = []
                for seg_idx, seg in enumerate(vad_segments, start=1):
                    if not isinstance(seg, (list, tuple)) or len(seg) != 2:
                        continue
                    seg_start = int(seg[0])
                    seg_end = int(seg[1])
                    if seg_end <= seg_start:
                        continue

                    seg_wav = tmp_dir / f"seg_{seg_idx:04d}.wav"
                    extract_segment_wav(audio_path, seg_start, seg_end, seg_wav)

                    asr_res = asr_model.generate(
                        input=str(seg_wav),
                        disable_pbar=True,
                    )
                    text = ""
                    if asr_res and isinstance(asr_res, list):
                        text = (asr_res[0].get("text") or "").strip()

                    if text:
                        punc_res = punc_model.generate(input=text, disable_pbar=True)
                        if punc_res and isinstance(punc_res, list):
                            text = (punc_res[0].get("text") or text).strip()

                    if text:
                        segments.append({"start_ms": seg_start, "end_ms": seg_end, "text": text})

                    per_seg_results.append(
                        {
                            "segment_index": seg_idx,
                            "start_ms": seg_start,
                            "end_ms": seg_end,
                            "text": text,
                        }
                    )

                item = {
                    "audio": str(audio_path),
                    "vad_segments": vad_segments,
                    "segments": per_seg_results,
                }
                write_outputs(sample_out, item, segments)

                mf.write(f"{sample_id} | {audio_path}\n")
                print(f"[OK] Done: {audio_path.name}, segments={len(segments)}")

            except Exception as e:
                failed += 1
                err_file = sample_out / "error.txt"
                sample_out.mkdir(parents=True, exist_ok=True)
                with err_file.open("w", encoding="utf-8") as ef:
                    ef.write(str(e) + "\n\n")
                    ef.write(traceback.format_exc())
                print(f"[ERROR] Failed: {audio_path.name}")
                print(f"[ERROR] {e}")

    print("\n[SUMMARY] Total:", len(audio_files), "Failed:", failed)
    print(f"[SUMMARY] Output dir: {output_dir}")
    print(f"[SUMMARY] ID map: {map_file}")
    return 3 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
