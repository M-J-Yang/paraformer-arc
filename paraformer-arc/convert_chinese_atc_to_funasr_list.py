#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert a wav/txt paired dataset into FunASR/Paraformer data/list format.

Output files:
- train_wav.scp
- train_text.txt
- val_wav.scp
- val_text.txt
Optional:
- train.jsonl
- val.jsonl

Expected dataset layout example:
dataset_root/
├── WAVdata/
│   └── .../*.wav
└── TXTdata/
    └── .../*.txt

Matching rule:
- wav stem matches txt stem
- optional suffix stripping for wav stem, e.g. ch03_01_568_16k.wav -> ch03_01_568
"""

from __future__ import annotations

import argparse
import json
import random
import re
import wave
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_utt_id(stem: str, strip_wav_suffix: str | None = None) -> str:
    if strip_wav_suffix and stem.endswith(strip_wav_suffix):
        return stem[: -len(strip_wav_suffix)]
    return stem


def get_wav_duration_frames(wav_path: Path) -> int:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes()
    except Exception:
        return -1


def build_wav_map(
    wav_dir: Path,
    strip_wav_suffix: str | None,
) -> Tuple[Dict[str, Path], List[str]]:
    wav_map: Dict[str, Path] = {}
    duplicate_keys: List[str] = []

    for p in sorted(wav_dir.rglob("*.wav")):
        key = normalize_utt_id(p.stem, strip_wav_suffix)

        if key not in wav_map:
            wav_map[key] = p
        else:
            duplicate_keys.append(key)
            # Prefer file whose original stem endswith strip suffix, e.g. *_16k.wav
            if strip_wav_suffix and p.stem.endswith(strip_wav_suffix):
                wav_map[key] = p

    return wav_map, sorted(set(duplicate_keys))


def build_txt_map(
    txt_dir: Path,
    txt_encoding: str,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    txt_map: Dict[str, str] = {}
    duplicate_keys: List[str] = []
    empty_text_keys: List[str] = []

    for p in sorted(txt_dir.rglob("*.txt")):
        # 跳过 txt 根目录下可能的说明文件
        if p.parent == txt_dir:
            continue

        key = p.stem
        try:
            text = p.read_text(encoding=txt_encoding, errors="replace")
        except Exception:
            text = p.read_text(encoding="utf-8", errors="replace")

        text = normalize_text(text)

        if not text:
            empty_text_keys.append(key)
            continue

        if key in txt_map:
            duplicate_keys.append(key)

        txt_map[key] = text

    return txt_map, sorted(set(duplicate_keys)), sorted(set(empty_text_keys))


def build_records(
    wav_map: Dict[str, Path],
    txt_map: Dict[str, str],
) -> Tuple[List[Tuple[str, Path, str]], List[str], List[str]]:
    common_keys = sorted(set(wav_map) & set(txt_map))
    missing_txt = sorted(set(wav_map) - set(txt_map))
    missing_wav = sorted(set(txt_map) - set(wav_map))

    records: List[Tuple[str, Path, str]] = []
    for key in common_keys:
        records.append((key, wav_map[key], txt_map[key]))

    return records, missing_txt, missing_wav


def to_output_wav_path(wav_path: Path, project_root: Path, path_mode: str) -> str:
    wav_abs = wav_path.resolve()
    if path_mode == "abs":
        return str(wav_abs)
    return str(wav_abs.relative_to(project_root.resolve())).replace("\\", "/")


def write_pairs(path: Path, rows: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for utt_id, value in rows:
            f.write(f"{utt_id} {value}\n")


def write_jsonl(path: Path, records: List[Tuple[str, Path, str]], project_root: Path, path_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for utt_id, wav_path, text in records:
            source_path = to_output_wav_path(wav_path, project_root, path_mode)
            source_len = get_wav_duration_frames(wav_path)
            item = {
                "key": utt_id,
                "source": source_path,
                "source_len": source_len,
                "target": text,
                "target_len": len(text),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_records(
    records: List[Tuple[str, Path, str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[str, Path, str]], List[Tuple[str, Path, str]]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_ratio))
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]
    return train_records, val_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert wav/txt dataset into Paraformer/FunASR list format")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root path")
    parser.add_argument("--wav-subdir", default="WAVdata", help="Wav subdirectory under dataset root")
    parser.add_argument("--txt-subdir", default="TXTdata", help="Txt subdirectory under dataset root")
    parser.add_argument("--output-dir", type=Path, default=Path("FunASR/data/list"), help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=20260308, help="Random seed")
    parser.add_argument("--txt-encoding", default="utf-8", help="Text encoding")
    parser.add_argument("--path-mode", choices=["abs", "rel"], default="abs", help="Path mode for wav.scp/jsonl")
    parser.add_argument(
        "--strip-wav-suffix",
        default="_16k",
        help="Strip wav stem suffix before matching txt stem, empty string to disable",
    )
    parser.add_argument(
        "--also-generate-jsonl",
        action="store_true",
        help="Also generate train.jsonl and val.jsonl directly",
    )
    args = parser.parse_args()

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1)")

    project_root = args.project_root.resolve()
    dataset_root = args.dataset_root.resolve() if args.dataset_root.is_absolute() else (project_root / args.dataset_root).resolve()
    output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (project_root / args.output_dir).resolve()

    wav_dir = dataset_root / args.wav_subdir
    txt_dir = dataset_root / args.txt_subdir

    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")
    if not txt_dir.exists():
        raise FileNotFoundError(f"TXT directory not found: {txt_dir}")

    strip_suffix = args.strip_wav_suffix.strip() or None

    wav_map, wav_dup = build_wav_map(wav_dir, strip_suffix)
    txt_map, txt_dup, empty_txt = build_txt_map(txt_dir, args.txt_encoding)

    records, missing_txt, missing_wav = build_records(wav_map, txt_map)

    if not records:
        raise RuntimeError("No valid wav/txt pairs found")

    train_records, val_records = split_records(records, args.val_ratio, args.seed)

    train_wav_rows = [(utt, to_output_wav_path(wav, project_root, args.path_mode)) for utt, wav, _ in train_records]
    val_wav_rows = [(utt, to_output_wav_path(wav, project_root, args.path_mode)) for utt, wav, _ in val_records]
    train_text_rows = [(utt, text) for utt, _, text in train_records]
    val_text_rows = [(utt, text) for utt, _, text in val_records]

    write_pairs(output_dir / "train_wav.scp", train_wav_rows)
    write_pairs(output_dir / "val_wav.scp", val_wav_rows)
    write_pairs(output_dir / "train_text.txt", train_text_rows)
    write_pairs(output_dir / "val_text.txt", val_text_rows)

    if args.also_generate_jsonl:
        write_jsonl(output_dir / "train.jsonl", train_records, project_root, args.path_mode)
        write_jsonl(output_dir / "val.jsonl", val_records, project_root, args.path_mode)

    print("=" * 60)
    print("Convert finished")
    print(f"Dataset root : {dataset_root}")
    print(f"WAV dir      : {wav_dir}")
    print(f"TXT dir      : {txt_dir}")
    print(f"Output dir   : {output_dir}")
    print(f"Total pairs  : {len(records)}")
    print(f"Train pairs  : {len(train_records)}")
    print(f"Val pairs    : {len(val_records)}")
    print(f"Path mode    : {args.path_mode}")
    print(f"Strip suffix : {strip_suffix}")
    print("-" * 60)
    print(f"Missing txt        : {len(missing_txt)}")
    print(f"Missing wav        : {len(missing_wav)}")
    print(f"Duplicate wav keys : {len(wav_dup)}")
    print(f"Duplicate txt keys : {len(txt_dup)}")
    print(f"Empty txt files    : {len(empty_txt)}")

    if missing_txt:
        print(f"Example missing txt: {missing_txt[:5]}")
    if missing_wav:
        print(f"Example missing wav: {missing_wav[:5]}")
    if wav_dup:
        print(f"Example dup wav key: {wav_dup[:5]}")
    if txt_dup:
        print(f"Example dup txt key: {txt_dup[:5]}")
    if empty_txt:
        print(f"Example empty txt  : {empty_txt[:5]}")

    print("=" * 60)


if __name__ == "__main__":
    main()