"""
测试集 CER 评估脚本
在测试集上评估 ASR 系统的字符错误率（Character Error Rate）
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import editdistance
from loguru import logger

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atc_asr_pipeline import ATCASRPipeline


def calculate_cer(reference: str, hypothesis: str) -> Tuple[float, int, int, int, int]:
    """
    计算字符错误率（CER）

    Args:
        reference: 参考文本（ground truth）
        hypothesis: 识别文本

    Returns:
        (cer, substitutions, deletions, insertions, total_chars)
    """
    # 移除空格进行字符级别比较
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    # 计算编辑距离
    distance = editdistance.eval(ref_chars, hyp_chars)

    # 总字符数
    total_chars = len(ref_chars)

    if total_chars == 0:
        return 0.0, 0, 0, 0, 0

    # CER = 编辑距离 / 参考文本长度
    cer = distance / total_chars

    # 详细的替换、删除、插入统计（简化版本）
    # 这里使用编辑距离作为总错误数
    return cer, distance, 0, 0, total_chars


def evaluate_test_set(
    test_samples: List[Dict[str, str]],
    pipeline: ATCASRPipeline,
    output_dir: Path
) -> Dict[str, any]:
    """
    在测试集上评估

    Args:
        test_samples: 测试集样本
        pipeline: ASR Pipeline
        output_dir: 输出目录

    Returns:
        评估结果字典
    """
    results = []
    total_cer = 0.0
    total_chars = 0
    total_errors = 0

    logger.info(f"开始评估测试集 ({len(test_samples)} 个样本)...")

    for sample in tqdm(test_samples, desc="评估进度"):
        audio_path = sample["audio"]
        ground_truth = sample["gt"]

        try:
            # ASR 识别
            result = pipeline.process(audio_path)

            # 计算 CER
            raw_text = result["raw_text"]
            final_text = result["final_text"]

            # 原始 ASR 的 CER
            raw_cer, raw_errors, _, _, chars = calculate_cer(ground_truth, raw_text)

            # 后处理后的 CER
            final_cer, final_errors, _, _, _ = calculate_cer(ground_truth, final_text)

            # 累计统计
            total_cer += final_cer
            total_chars += chars
            total_errors += final_errors

            # 保存结果
            results.append({
                "audio_file": Path(audio_path).name,
                "ground_truth": ground_truth,
                "raw_text": raw_text,
                "final_text": final_text,
                "raw_cer": raw_cer,
                "final_cer": final_cer,
                "chars": chars,
                "errors": final_errors
            })

        except Exception as e:
            logger.error(f"处理失败 {audio_path}: {e}")
            results.append({
                "audio_file": Path(audio_path).name,
                "ground_truth": ground_truth,
                "error": str(e)
            })

    # 计算平均 CER
    avg_cer = total_cer / len(test_samples) if test_samples else 0.0

    # 汇总统计
    summary = {
        "total_samples": len(test_samples),
        "total_chars": total_chars,
        "total_errors": total_errors,
        "average_cer": avg_cer,
        "accuracy": 1.0 - avg_cer
    }

    # 保存详细结果
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "test_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_dir / "test_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"评估结果已保存到: {output_dir}")

    return summary, results


def generate_report(summary: Dict, results: List[Dict], output_path: Path):
    """
    生成评估报告

    Args:
        summary: 汇总统计
        results: 详细结果
        output_path: 报告输出路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ATC 语音识别系统 - 测试集评估报告\n")
        f.write("=" * 80 + "\n\n")

        # 总体统计
        f.write("## 总体统计\n\n")
        f.write(f"测试样本数:   {summary['total_samples']}\n")
        f.write(f"总字符数:     {summary['total_chars']}\n")
        f.write(f"总错误数:     {summary['total_errors']}\n")
        f.write(f"平均 CER:     {summary['average_cer']:.4f} ({summary['average_cer']*100:.2f}%)\n")
        f.write(f"识别准确率:   {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)\n")
        f.write("\n")

        # CER 分布统计
        f.write("## CER 分布\n\n")
        cer_ranges = {
            "完美 (CER = 0)": 0,
            "优秀 (0 < CER ≤ 0.05)": 0,
            "良好 (0.05 < CER ≤ 0.10)": 0,
            "一般 (0.10 < CER ≤ 0.20)": 0,
            "较差 (CER > 0.20)": 0
        }

        for result in results:
            if "final_cer" in result:
                cer = result["final_cer"]
                if cer == 0:
                    cer_ranges["完美 (CER = 0)"] += 1
                elif cer <= 0.05:
                    cer_ranges["优秀 (0 < CER ≤ 0.05)"] += 1
                elif cer <= 0.10:
                    cer_ranges["良好 (0.05 < CER ≤ 0.10)"] += 1
                elif cer <= 0.20:
                    cer_ranges["一般 (0.10 < CER ≤ 0.20)"] += 1
                else:
                    cer_ranges["较差 (CER > 0.20)"] += 1

        for range_name, count in cer_ranges.items():
            percentage = count / summary['total_samples'] * 100 if summary['total_samples'] > 0 else 0
            f.write(f"{range_name}: {count} ({percentage:.1f}%)\n")

        f.write("\n")

        # 最佳识别样本
        f.write("## 最佳识别样本 (Top 10)\n\n")
        sorted_results = sorted([r for r in results if "final_cer" in r], key=lambda x: x["final_cer"])
        for i, result in enumerate(sorted_results[:10], 1):
            f.write(f"[{i}] {result['audio_file']} (CER: {result['final_cer']:.4f})\n")
            f.write(f"    GT:   {result['ground_truth']}\n")
            f.write(f"    识别: {result['final_text']}\n\n")

        # 最差识别样本
        f.write("## 最差识别样本 (Top 10)\n\n")
        worst_results = sorted([r for r in results if "final_cer" in r], key=lambda x: x["final_cer"], reverse=True)
        for i, result in enumerate(worst_results[:10], 1):
            f.write(f"[{i}] {result['audio_file']} (CER: {result['final_cer']:.4f})\n")
            f.write(f"    GT:   {result['ground_truth']}\n")
            f.write(f"    识别: {result['final_text']}\n\n")

        f.write("=" * 80 + "\n")

    logger.info(f"评估报告已生成: {output_path}")


def main():
    """主函数"""
    # 加载测试集
    test_file = Path("D:/NPU_works/语音/ATC-paraformer/data_splits/test.json")

    if not test_file.exists():
        logger.error(f"测试集文件不存在: {test_file}")
        logger.info("请先运行 split_dataset.py 划分数据集")
        return

    with open(test_file, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)

    logger.info(f"加载测试集: {len(test_samples)} 个样本")

    # 初始化 Pipeline
    logger.info("初始化 ASR Pipeline...")
    pipeline = ATCASRPipeline(device="cuda:0", use_hotwords=True)

    # 评估测试集
    output_dir = Path("D:/NPU_works/语音/ATC-paraformer/evaluation")
    summary, results = evaluate_test_set(test_samples, pipeline, output_dir)

    # 生成报告
    report_path = output_dir / "evaluation_report.txt"
    generate_report(summary, results, report_path)

    # 打印摘要
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)
    print(f"测试样本数:   {summary['total_samples']}")
    print(f"平均 CER:     {summary['average_cer']:.4f} ({summary['average_cer']*100:.2f}%)")
    print(f"识别准确率:   {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)")
    print("=" * 80)
    print(f"\n详细报告: {report_path}")


if __name__ == "__main__":
    main()
