"""
微调模型推理评估脚本
使用 finetune/output/model.pt 在 data_splits/test.json 上评估 CER
"""
import json
import sys
from pathlib import Path

import editdistance
from tqdm import tqdm
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
PRETRAINED_MODEL_DIR = PROJECT_ROOT / "finetune" / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
FINETUNED_OUTPUT_DIR = PROJECT_ROOT / "finetune" / "output"
TEST_JSON = PROJECT_ROOT / "data_splits" / "test.json"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "finetuned"

# 旧路径前缀 → 新路径前缀（data_splits 里音频路径用的是旧目录名）
OLD_PREFIX = "D:\\NPU_works\\语音\\ATC-paraformer"
NEW_PREFIX = str(PROJECT_ROOT).replace("/", "\\")


def fix_audio_path(path: str) -> str:
    """修正 test.json 里的旧路径"""
    if OLD_PREFIX in path:
        return path.replace(OLD_PREFIX, NEW_PREFIX)
    return path


def load_model():
    """加载微调后的模型

    FunASR AutoModel.build_model 只处理 init_param，不会自动加载 model.pt。
    正确流程:
    1. 用 output/config.yaml（移除 init_param）让 FunASR 建出 12 层 Paraformer 架构
    2. 手动加载 model.pt.avg3 state_dict → 完全替换随机权重
    """
    try:
        from funasr import AutoModel
    except ImportError:
        logger.error("FunASR 未安装，请运行: pip install funasr")
        sys.exit(1)

    import shutil, yaml, torch

    # 优先用 avg3（3 个最佳 epoch 平均）
    model_pt = None
    for candidate in ["model.pt.avg3", "model.pt.best", "model.pt"]:
        path = FINETUNED_OUTPUT_DIR / candidate
        if path.exists():
            model_pt = path
            break
    if model_pt is None:
        logger.error(f"未找到微调模型权重: {FINETUNED_OUTPUT_DIR}")
        sys.exit(1)
    logger.info(f"使用微调权重: {model_pt.name}")

    config_yaml = FINETUNED_OUTPUT_DIR / "config.yaml"
    config_yaml_bak = FINETUNED_OUTPUT_DIR / "_config.yaml.bak"

    # 读训练 config，移除 init_param（避免 FunASR 在 build_model 时加载 seaco 权重）
    with open(config_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.pop("init_param", None)
    cfg.pop("train_data_set_list", None)
    cfg.pop("valid_data_set_list", None)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"推理设备: {device}")

    # 临时替换 config.yaml（移除 init_param，FunASR 将建出 12 层 Paraformer 随机初始化）
    shutil.copy2(config_yaml, config_yaml_bak)
    with open(config_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    try:
        logger.info("Step 1: 建出 12 层 Paraformer 架构 (随机初始化，无 init_param)")
        model = AutoModel(model=str(FINETUNED_OUTPUT_DIR), device=device, disable_update=True)
    finally:
        shutil.copy2(config_yaml_bak, config_yaml)
        config_yaml_bak.unlink()

    # 确认内部模型类型
    inner_model = model.model
    logger.info(f"内部模型类型: {inner_model.__class__.__name__}")

    # Step 2: 手动加载微调 state_dict
    logger.info(f"Step 2: 加载微调 state_dict: {model_pt.name}")
    ckpt = torch.load(str(model_pt), map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    current_sd = inner_model.state_dict()
    # 过滤掉形状不匹配的 key（以防万一）
    filtered_sd = {}
    shape_mismatches = []
    for k, v in state_dict.items():
        if k in current_sd:
            if v.shape == current_sd[k].shape:
                filtered_sd[k] = v
            else:
                shape_mismatches.append((k, v.shape, current_sd[k].shape))
        # 不在当前模型的 key → 忽略（unexpected）

    if shape_mismatches:
        logger.warning(f"形状不匹配（跳过 {len(shape_mismatches)} 个 key）:")
        for k, src_shape, dst_shape in shape_mismatches[:5]:
            logger.warning(f"  {k}: ckpt={src_shape} model={dst_shape}")

    missing_after = set(current_sd.keys()) - set(filtered_sd.keys())
    logger.info(f"  加载 {len(filtered_sd)}/{len(current_sd)} 个参数"
                f"  (跳过形状不匹配={len(shape_mismatches)}, 未覆盖={len(missing_after)})")

    result = inner_model.load_state_dict(filtered_sd, strict=False)
    logger.info(f"  load_state_dict 结果 - missing: {len(result.missing_keys)}, unexpected: {len(result.unexpected_keys)}")
    logger.info("微调模型加载完成")

    return model


def calculate_cer(reference: str, hypothesis: str):
    """计算字符错误率"""
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    distance = editdistance.eval(ref_chars, hyp_chars)
    total = len(ref_chars)
    cer = distance / total if total > 0 else 0.0
    return cer, distance, total


def main():
    logger.info("=" * 70)
    logger.info("微调模型评估 — ATC 测试集")
    logger.info("=" * 70)

    # 加载测试集
    if not TEST_JSON.exists():
        logger.error(f"测试集不存在: {TEST_JSON}")
        sys.exit(1)

    with open(TEST_JSON, "r", encoding="utf-8") as f:
        test_samples = json.load(f)
    logger.info(f"测试集样本数: {len(test_samples)}")

    # 修正路径并过滤不存在的文件
    valid_samples = []
    skipped = 0
    for s in test_samples:
        audio = fix_audio_path(s["audio"])
        if Path(audio).exists():
            valid_samples.append({"audio": audio, "gt": s["gt"]})
        else:
            skipped += 1
    if skipped:
        logger.warning(f"跳过 {skipped} 个找不到音频文件的样本")
    logger.info(f"有效样本数: {len(valid_samples)}")

    # 加载模型
    model = load_model()

    # 逐样本推理
    results = []
    total_errors = 0
    total_chars = 0

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(valid_samples, desc="推理进度"):
        audio_path = sample["audio"]
        gt = sample["gt"]
        try:
            raw = model.generate(input=audio_path, batch_size=1)
            # 提取文本
            if isinstance(raw, list) and raw:
                hyp = raw[0].get("text", "") if isinstance(raw[0], dict) else str(raw[0])
            elif isinstance(raw, dict):
                hyp = raw.get("text", "")
            else:
                hyp = str(raw)

            cer, errors, chars = calculate_cer(gt, hyp)
            total_errors += errors
            total_chars += chars

            results.append({
                "audio": Path(audio_path).name,
                "gt": gt,
                "hyp": hyp,
                "cer": round(cer, 4),
                "errors": errors,
                "chars": chars,
            })
        except Exception as e:
            logger.warning(f"推理失败 [{Path(audio_path).name}]: {e}")
            results.append({
                "audio": Path(audio_path).name,
                "gt": gt,
                "hyp": "",
                "error": str(e),
            })

    # 汇总统计
    valid_results = [r for r in results if "cer" in r]
    avg_cer = sum(r["cer"] for r in valid_results) / len(valid_results) if valid_results else 0.0
    global_cer = total_errors / total_chars if total_chars > 0 else 0.0

    # CER 分布
    dist = {"完美(=0)": 0, "优秀(≤5%)": 0, "良好(≤10%)": 0, "一般(≤20%)": 0, "较差(>20%)": 0}
    for r in valid_results:
        c = r["cer"]
        if c == 0:
            dist["完美(=0)"] += 1
        elif c <= 0.05:
            dist["优秀(≤5%)"] += 1
        elif c <= 0.10:
            dist["良好(≤10%)"] += 1
        elif c <= 0.20:
            dist["一般(≤20%)"] += 1
        else:
            dist["较差(>20%)"] += 1

    summary = {
        "total_samples": len(valid_samples),
        "valid_results": len(valid_results),
        "total_chars": total_chars,
        "total_errors": total_errors,
        "global_cer": round(global_cer, 4),
        "avg_cer": round(avg_cer, 4),
        "accuracy": round(1.0 - avg_cer, 4),
        "cer_distribution": dist,
    }

    # 保存结果
    with open(EVAL_OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(EVAL_OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印报告
    n = len(valid_results)
    print("\n" + "=" * 70)
    print("微调模型评估结果")
    print("=" * 70)
    print(f"有效样本数:    {n}")
    print(f"总字符数:      {total_chars}")
    print(f"总错误字符:    {total_errors}")
    print(f"全局 CER:      {global_cer*100:.2f}%  (总错误/总字符)")
    print(f"平均 CER:      {avg_cer*100:.2f}%  (各样本CER均值)")
    print(f"字符识别率:    {(1-avg_cer)*100:.2f}%")
    print()
    print("CER 分布:")
    for k, v in dist.items():
        print(f"  {k}: {v} ({v/n*100:.1f}%)")

    # Top 5 最差样本
    worst = sorted(valid_results, key=lambda x: x["cer"], reverse=True)[:5]
    print("\nTop 5 最差样本:")
    for i, r in enumerate(worst, 1):
        print(f"  [{i}] {r['audio']}  CER={r['cer']*100:.1f}%")
        print(f"       GT:  {r['gt']}")
        print(f"       识别: {r['hyp']}")

    # Top 5 最佳样本
    best = sorted(valid_results, key=lambda x: x["cer"])[:5]
    print("\nTop 5 最佳样本:")
    for i, r in enumerate(best, 1):
        print(f"  [{i}] {r['audio']}  CER={r['cer']*100:.1f}%")
        print(f"       GT:  {r['gt']}")
        print(f"       识别: {r['hyp']}")

    print("=" * 70)
    print(f"详细结果已保存: {EVAL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
