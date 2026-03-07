"""
数据集划分脚本
将 ATC 数据集按照 8:1:1 划分为训练集、验证集、测试集
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from loguru import logger


def collect_audio_files(data_dir: Path) -> List[Dict[str, str]]:
    """
    收集所有音频文件及其对应的文本文件

    Returns:
        List of {"audio": audio_path, "text": text_path, "gt": ground_truth}
    """
    audio_dir = data_dir / "WAVdata"
    text_dir = data_dir / "TXTdata"

    samples = []

    # 遍历所有 WAV 文件
    for audio_path in audio_dir.glob("**/*.wav"):
        # 构建对应的文本路径
        relative_path = audio_path.relative_to(audio_dir)
        text_path = text_dir / relative_path.parent / (relative_path.stem + ".txt")

        if text_path.exists():
            # 读取 ground truth
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # 移除行号前缀
                    if '→' in content:
                        gt = content.split('→', 1)[1]
                    else:
                        gt = content

                samples.append({
                    "audio": str(audio_path),
                    "text": str(text_path),
                    "gt": gt
                })
            except Exception as e:
                logger.warning(f"读取文本失败 {text_path}: {e}")
        else:
            logger.warning(f"文本文件不存在: {text_path}")

    return samples


def split_dataset(
    samples: List[Dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    划分数据集

    Args:
        samples: 所有样本
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        (train_samples, val_samples, test_samples)
    """
    # 设置随机种子
    random.seed(seed)

    # 打乱数据
    samples_shuffled = samples.copy()
    random.shuffle(samples_shuffled)

    # 计算划分点
    total = len(samples_shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_samples = samples_shuffled[:train_end]
    val_samples = samples_shuffled[train_end:val_end]
    test_samples = samples_shuffled[val_end:]

    logger.info(f"数据集划分完成:")
    logger.info(f"  训练集: {len(train_samples)} ({len(train_samples)/total*100:.1f}%)")
    logger.info(f"  验证集: {len(val_samples)} ({len(val_samples)/total*100:.1f}%)")
    logger.info(f"  测试集: {len(test_samples)} ({len(test_samples)/total*100:.1f}%)")

    return train_samples, val_samples, test_samples


def save_split(
    train_samples: List[Dict[str, str]],
    val_samples: List[Dict[str, str]],
    test_samples: List[Dict[str, str]],
    output_dir: Path
):
    """保存划分结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存训练集
    with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)

    # 保存验证集
    with open(output_dir / "val.json", 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)

    # 保存测试集
    with open(output_dir / "test.json", 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)

    logger.info(f"数据集划分文件已保存到: {output_dir}")


def main():
    """主函数"""
    # 数据目录
    data_dir = Path("D:/NPU_works/语音/ATC-paraformer/chinese_ATC_formatted")
    output_dir = Path("D:/NPU_works/语音/ATC-paraformer/data_splits")

    logger.info("开始收集数据集...")
    samples = collect_audio_files(data_dir)
    logger.info(f"共收集到 {len(samples)} 个样本")

    if len(samples) == 0:
        logger.error("未找到任何样本！")
        return

    # 划分数据集 (8:1:1)
    logger.info("开始划分数据集 (8:1:1)...")
    train_samples, val_samples, test_samples = split_dataset(
        samples,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )

    # 保存划分结果
    save_split(train_samples, val_samples, test_samples, output_dir)

    # 打印统计信息
    print("\n" + "=" * 60)
    print("数据集划分统计")
    print("=" * 60)
    print(f"总样本数: {len(samples)}")
    print(f"训练集:   {len(train_samples)} ({len(train_samples)/len(samples)*100:.1f}%)")
    print(f"验证集:   {len(val_samples)} ({len(val_samples)/len(samples)*100:.1f}%)")
    print(f"测试集:   {len(test_samples)} ({len(test_samples)/len(samples)*100:.1f}%)")
    print("=" * 60)

    # 显示部分样本
    print("\n训练集样本示例:")
    for i, sample in enumerate(train_samples[:3], 1):
        print(f"  [{i}] {Path(sample['audio']).name}: {sample['gt']}")

    print("\n测试集样本示例:")
    for i, sample in enumerate(test_samples[:3], 1):
        print(f"  [{i}] {Path(sample['audio']).name}: {sample['gt']}")


if __name__ == "__main__":
    main()
