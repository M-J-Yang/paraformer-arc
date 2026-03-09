import argparse
import json
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


def to_ms(v):
    try:
        return int(round(float(v)))
    except Exception:
        return 0


def normalize_segments(item: dict):
    segments = []
    sentence_info = item.get("sentence_info") or []
    for sent in sentence_info:
        text = (sent.get("text") or sent.get("sentence") or "").strip()
        start_ms = to_ms(sent.get("start", 0))
        end_ms = to_ms(sent.get("end", 0))
        if text:
            segments.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})

    if not segments:
        text = (item.get("text") or "").strip()
        ts = item.get("timestamp") or []
        if text and ts:
            segments.append(
                {
                    "start_ms": to_ms(ts[0][0]),
                    "end_ms": to_ms(ts[-1][1]),
                    "text": text,
                }
            )
        elif text:
            segments.append({"start_ms": 0, "end_ms": 0, "text": text})
    return segments


def write_outputs(out_dir: Path, result_item: dict, segments: list):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result_item, f, ensure_ascii=False, indent=2)

    with (out_dir / "segments.txt").open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg['start_ms']/1000.0:.3f}\t{seg['end_ms']/1000.0:.3f}\t{seg['text']}\n")

    merged = "".join([s["text"] for s in segments]).strip()
    with (out_dir / "text.txt").open("w", encoding="utf-8") as f:
        f.write(merged + "\n")


def check_local_model_dir(path: Path, name: str):
    if not (path / "config.yaml").exists():
        raise FileNotFoundError(f"{name} config not found: {path / 'config.yaml'}")
    if not (path / "model.pt").exists():
        raise FileNotFoundError(f"{name} weights not found: {path / 'model.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Timestamp-predictor ASR inference (local models only)")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--vad-model", type=str, required=True)
    parser.add_argument("--punc-model", type=str, required=True)
    parser.add_argument("--force-device", type=str, default="")
    parser.add_argument("--max-single-segment-time", type=int, default=30000)
    parser.add_argument("--batch-size-s", type=int, default=20)
    parser.add_argument("--batch-size-threshold-s", type=int, default=10)
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    vad_model_dir = Path(args.vad_model).resolve()
    punc_model_dir = Path(args.punc_model).resolve()

    check_local_model_dir(model_dir, "ASR model")
    check_local_model_dir(vad_model_dir, "VAD model")
    check_local_model_dir(punc_model_dir, "PUNC model")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio folder not found: {audio_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    map_file = output_dir / "file_map.txt"

    device = choose_device(args.force_device)
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] ASR model: {model_dir}")
    print(f"[INFO] VAD model: {vad_model_dir}")
    print(f"[INFO] PUNC model: {punc_model_dir}")

    model = AutoModel(
        model=str(model_dir),
        init_param=str(model_dir / "model.pt"),
        tokenizer_conf={
            "token_list": str(model_dir / "tokens.json"),
            "seg_dict_file": str(model_dir / "seg_dict"),
        },
        frontend_conf={"cmvn_file": str(model_dir / "am.mvn")},
        vad_model=str(vad_model_dir),
        punc_model=str(punc_model_dir),
        vad_kwargs={"max_single_segment_time": args.max_single_segment_time},
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
                res = model.generate(
                    input=str(audio_path),
                    sentence_timestamp=True,
                    return_raw_text=True,
                    batch_size_s=args.batch_size_s,
                    batch_size_threshold_s=args.batch_size_threshold_s,
                    disable_pbar=True,
                )
                if not res:
                    raise RuntimeError("Empty inference result")
                item = res[0]

                if ("timestamp" not in item) and ("sentence_info" not in item):
                    raise RuntimeError(
                        "Model does not output timestamp fields. "
                        "Please use an ASR model trained with timestamp predictor."
                    )

                segments = normalize_segments(item)
                write_outputs(sample_out, item, segments)
                mf.write(f"{sample_id} | {audio_path}\n")
                print(f"[OK] Done: {audio_path.name}, segments={len(segments)}")
            except Exception as e:
                failed += 1
                sample_out.mkdir(parents=True, exist_ok=True)
                with (sample_out / "error.txt").open("w", encoding="utf-8") as ef:
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

