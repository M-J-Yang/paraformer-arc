"""
批量推理示例
演示如何批量处理音频文件
"""
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atc_asr_pipeline import ATCASRPipeline
from utils.config import Config
from loguru import logger
import json


def main():
    """批量推理主函数"""
    # 初始化 Pipeline
    logger.info("初始化 ATC ASR Pipeline")
    pipeline = ATCASRPipeline(device="cuda:0", use_hotwords=True)

    # 获取测试音频
    config = Config()
    audio_dir = config.AUDIO_DIR

    # 查找所有 WAV 文件
    audio_files = list(audio_dir.glob("**/*.wav"))[:10]  # 处理前 10 个

    if not audio_files:
        logger.error("未找到音频文件")
        return

    logger.info(f"找到 {len(audio_files)} 个音频文件")

    # 批量处理
    results = pipeline.process_batch(audio_files)

    # 保存结果
    output_file = Path(__file__).parent.parent / "output" / "batch_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已保存到: {output_file}")

    # 打印部分结果
    print("\n=== 批量推理结果 ===\n")
    for i, result in enumerate(results[:3], 1):
        print(f"[{i}] {Path(result['audio_file']).name}")
        print(f"    GT:     {result.get('ground_truth', 'N/A')}")
        print(f"    原始:   {result['raw_text']}")
        print(f"    标准化: {result['normalized_text']}")
        print(f"    最终:   {result['final_text']}")
        print(f"    呼号:   {result['callsign']}")
        print(f"    指令:   {result['commands']}")
        print()


if __name__ == "__main__":
    main()
