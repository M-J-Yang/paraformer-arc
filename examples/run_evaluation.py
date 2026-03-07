"""
完整评估流程脚本
1. 划分数据集 (8:1:1)
2. 在测试集上评估 CER
3. 生成评估报告
"""
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from split_dataset import collect_audio_files, split_dataset, save_split
from evaluate_test import evaluate_test_set, generate_report
from atc_asr_pipeline import ATCASRPipeline
from loguru import logger
import json


def main():
    """主函数"""
    # 路径配置
    data_dir = Path("D:/NPU_works/语音/ATC-paraformer/chinese_ATC_formatted")
    split_dir = Path("D:/NPU_works/语音/ATC-paraformer/data_splits")
    eval_dir = Path("D:/NPU_works/语音/ATC-paraformer/evaluation")

    print("\n" + "=" * 80)
    print("ATC 语音识别系统 - 完整评估流程")
    print("=" * 80 + "\n")

    # Step 1: 划分数据集
    print("Step 1: 划分数据集 (8:1:1)")
    print("-" * 80)

    # 检查是否已经划分过
    if (split_dir / "test.json").exists():
        logger.info("数据集已划分，直接加载...")
        with open(split_dir / "train.json", 'r', encoding='utf-8') as f:
            train_samples = json.load(f)
        with open(split_dir / "val.json", 'r', encoding='utf-8') as f:
            val_samples = json.load(f)
        with open(split_dir / "test.json", 'r', encoding='utf-8') as f:
            test_samples = json.load(f)
    else:
        logger.info("开始收集数据集...")
        samples = collect_audio_files(data_dir)
        logger.info(f"共收集到 {len(samples)} 个样本")

        logger.info("开始划分数据集...")
        train_samples, val_samples, test_samples = split_dataset(
            samples,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42
        )

        save_split(train_samples, val_samples, test_samples, split_dir)

    print(f"训练集: {len(train_samples)} 样本")
    print(f"验证集: {len(val_samples)} 样本")
    print(f"测试集: {len(test_samples)} 样本")
    print()

    # Step 2: 初始化 ASR Pipeline
    print("Step 2: 初始化 ASR Pipeline")
    print("-" * 80)
    pipeline = ATCASRPipeline(device="cuda:0", use_hotwords=True)
    print()

    # Step 3: 在测试集上评估
    print("Step 3: 在测试集上评估 CER")
    print("-" * 80)
    summary, results = evaluate_test_set(test_samples, pipeline, eval_dir)
    print()

    # Step 4: 生成报告
    print("Step 4: 生成评估报告")
    print("-" * 80)
    report_path = eval_dir / "evaluation_report.txt"
    generate_report(summary, results, report_path)
    print(f"评估报告已保存: {report_path}")
    print()

    # 打印汇总结果
    print("=" * 80)
    print("评估结果汇总")
    print("=" * 80)
    print(f"测试样本数:   {summary['total_samples']}")
    print(f"总字符数:     {summary['total_chars']}")
    print(f"总错误数:     {summary['total_errors']}")
    print(f"平均 CER:     {summary['average_cer']:.4f} ({summary['average_cer']*100:.2f}%)")
    print(f"识别准确率:   {summary['accuracy']:.4f} ({summary['accuracy']*100:.2f}%)")
    print("=" * 80)
    print()

    print(f"详细结果已保存到: {eval_dir}")
    print(f"  - test_results.json: 每个样本的详细结果")
    print(f"  - test_summary.json: 汇总统计")
    print(f"  - evaluation_report.txt: 完整评估报告")


if __name__ == "__main__":
    main()
