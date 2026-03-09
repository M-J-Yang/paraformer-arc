#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

from funasr import AutoModel


def edit_distance(ref, hyp):
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]


def calc_cer(ref: str, hyp: str):
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return edit_distance(list(ref), list(hyp)) / len(ref)


def normalize_text(s: str) -> str:
    return "".join(str(s).strip().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-jsonl", required=True, help="Path to val.jsonl")
    parser.add_argument("--model", required=True, help="Model dir or model hub id")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 / cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    val_jsonl = Path(args.val_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    with val_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from {val_jsonl}")

    model = AutoModel(
        model=args.model,
        device=args.device,
    )

    total_chars = 0
    total_errs = 0
    results = []

    for idx, item in enumerate(samples, 1):
        key = item["key"]
        wav_path = item["source"]
        ref = normalize_text(item["target"])

        res = model.generate(input=wav_path, batch_size_s=args.batch_size)
        if isinstance(res, list) and len(res) > 0:
            pred = res[0].get("text", "")
        elif isinstance(res, dict):
            pred = res.get("text", "")
        else:
            pred = ""

        hyp = normalize_text(pred)
        cer = calc_cer(ref, hyp)

        total_chars += len(ref)
        total_errs += edit_distance(list(ref), list(hyp))

        results.append({
            "key": key,
            "wav": wav_path,
            "ref": ref,
            "hyp": hyp,
            "cer": cer,
        })

        print(f"[{idx}/{len(samples)}] {key} CER={cer:.4f} REF={ref} HYP={hyp}")

    overall_cer = total_errs / total_chars if total_chars > 0 else 0.0

    result_json = output_dir / "val_cer_results.jsonl"
    with result_json.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    result_txt = output_dir / "val_cer_summary.txt"
    with result_txt.open("w", encoding="utf-8") as f:
        f.write(f"num_samples={len(results)}\n")
        f.write(f"total_chars={total_chars}\n")
        f.write(f"total_errs={total_errs}\n")
        f.write(f"CER={overall_cer:.6f}\n")

    print("=" * 60)
    print(f"Validation CER: {overall_cer:.6f}")
    print(f"Detailed results: {result_json}")
    print(f"Summary: {result_txt}")
    print("=" * 60)


if __name__ == "__main__":
    main()