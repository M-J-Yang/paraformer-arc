"""
FunASR Paraformer 微调脚本
在 ATC 数据集上微调预训练的 Paraformer 模型
"""
import os
import json
from pathlib import Path
from typing import Dict, List
import torch
from loguru import logger

try:
    from funasr import AutoModel
    from modelscope.trainers import build_trainer
    from modelscope.msdatasets import MsDataset
    from modelscope.utils.config import Config as MsConfig
except ImportError:
    logger.error("请安装 FunASR 和 ModelScope")
    raise


def prepare_finetune_data(
    train_samples: List[Dict],
    val_samples: List[Dict],
    output_dir: Path
):
    """
    准备微调数据格式

    FunASR 需要的格式:
    {
        "audio_filepath": "path/to/audio.wav",
        "text": "转录文本",
        "duration": 2.5
    }
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def convert_sample(sample: Dict) -> Dict:
        """转换单个样本"""
        return {
            "audio_filepath": sample["audio"],
            "text": sample["gt"],
            "duration": 0.0  # 可选，会自动计算
        }

    # 转换训练集
    train_data = [convert_sample(s) for s in train_samples]
    with open(output_dir / "train.jsonl", 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 转换验证集
    val_data = [convert_sample(s) for s in val_samples]
    with open(output_dir / "val.jsonl", 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"训练数据: {len(train_data)} 样本")
    logger.info(f"验证数据: {len(val_data)} 样本")
    logger.info(f"数据已保存到: {output_dir}")


def create_finetune_config(
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    base_model: str = "paraformer-zh",
    batch_size: int = 4,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
    warmup_steps: int = 1000,
) -> Dict:
    """
    创建微调配置
    """
    config = {
        "task": "auto-speech-recognition",
        "model": {
            "type": base_model,
            "model_revision": "v2.0.4"
        },
        "train": {
            "data_path": train_data_path,
            "batch_size": batch_size,
            "num_workers": 4,
            "shuffle": True,
        },
        "validation": {
            "data_path": val_data_path,
            "batch_size": batch_size,
            "num_workers": 2,
        },
        "trainer": {
            "max_epochs": num_epochs,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "gradient_clip_val": 5.0,
            "accumulate_grad_batches": 1,
            "val_check_interval": 0.5,  # 每半个 epoch 验证一次
            "save_top_k": 3,
            "monitor": "val_loss",
            "mode": "min",
        },
        "output_dir": output_dir,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    return config


def finetune_paraformer(
    train_jsonl: Path,
    val_jsonl: Path,
    output_dir: Path,
    base_model: str = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    batch_size: int = 4,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
):
    """
    微调 Paraformer 模型

    Args:
        train_jsonl: 训练数据路径
        val_jsonl: 验证数据路径
        output_dir: 输出目录
        base_model: 基础模型
        batch_size: 批量大小
        num_epochs: 训练轮数
        learning_rate: 学习率
    """
    logger.info("=" * 80)
    logger.info("开始微调 Paraformer 模型")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建配置
    logger.info("创建训练配置...")
    config = create_finetune_config(
        train_data_path=str(train_jsonl),
        val_data_path=str(val_jsonl),
        output_dir=str(output_dir),
        base_model=base_model,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )

    # 保存配置
    config_path = output_dir / "finetune_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"配置已保存: {config_path}")

    # 打印训练参数
    logger.info("\n训练参数:")
    logger.info(f"  基础模型: {base_model}")
    logger.info(f"  批量大小: {batch_size}")
    logger.info(f"  训练轮数: {num_epochs}")
    logger.info(f"  学习率: {learning_rate}")
    logger.info(f"  设备: {config['device']}")
    logger.info(f"  输出目录: {output_dir}")

    try:
        # 使用 FunASR 的微调接口
        logger.info("\n开始训练...")

        # 方法1: 使用 ModelScope Trainer (推荐)
        from modelscope.trainers import build_trainer
        from modelscope.msdatasets import MsDataset

        # 加载数据集
        train_dataset = MsDataset.load(
            str(train_jsonl),
            split='train',
            data_format='jsonl'
        )

        val_dataset = MsDataset.load(
            str(val_jsonl),
            split='validation',
            data_format='jsonl'
        )

        # 构建训练器
        trainer = build_trainer(
            name='speech-recognition-trainer',
            default_args={
                'model': base_model,
                'train_dataset': train_dataset,
                'eval_dataset': val_dataset,
                'work_dir': str(output_dir),
                'max_epochs': num_epochs,
                'lr': learning_rate,
                'batch_size': batch_size,
            }
        )

        # 开始训练
        trainer.train()

        logger.info("\n训练完成！")
        logger.info(f"模型已保存到: {output_dir}")

    except Exception as e:
        logger.error(f"训练失败: {e}")
        logger.info("\n备选方案: 使用 FunASR 命令行工具")
        logger.info("请运行以下命令:")
        logger.info(f"funasr-train \\")
        logger.info(f"  --model {base_model} \\")
        logger.info(f"  --train-data {train_jsonl} \\")
        logger.info(f"  --val-data {val_jsonl} \\")
        logger.info(f"  --output-dir {output_dir} \\")
        logger.info(f"  --batch-size {batch_size} \\")
        logger.info(f"  --num-epochs {num_epochs} \\")
        logger.info(f"  --learning-rate {learning_rate}")
        raise


def main():
    """主函数"""
    # 路径配置
    project_root = Path("D:/NPU_works/语音/ATC-paraformer")
    split_dir = project_root / "data_splits"
    finetune_data_dir = project_root / "finetune_data"
    output_dir = project_root / "finetuned_model"

    print("\n" + "=" * 80)
    print("Paraformer 微调流程")
    print("=" * 80 + "\n")

    # Step 1: 加载划分好的数据集
    print("Step 1: 加载数据集")
    print("-" * 80)

    with open(split_dir / "train.json", 'r', encoding='utf-8') as f:
        train_samples = json.load(f)

    with open(split_dir / "val.json", 'r', encoding='utf-8') as f:
        val_samples = json.load(f)

    print(f"训练集: {len(train_samples)} 样本")
    print(f"验证集: {len(val_samples)} 样本")
    print()

    # Step 2: 准备微调数据格式
    print("Step 2: 准备微调数据格式")
    print("-" * 80)
    prepare_finetune_data(train_samples, val_samples, finetune_data_dir)
    print()

    # Step 3: 微调模型
    print("Step 3: 微调 Paraformer 模型")
    print("-" * 80)

    finetune_paraformer(
        train_jsonl=finetune_data_dir / "train.jsonl",
        val_jsonl=finetune_data_dir / "val.jsonl",
        output_dir=output_dir,
        batch_size=4,  # 根据 GPU 显存调整
        num_epochs=20,
        learning_rate=1e-4,
    )

    print("\n" + "=" * 80)
    print("微调完成！")
    print("=" * 80)
    print(f"\n微调后的模型保存在: {output_dir}")
    print("\n下一步:")
    print("1. 使用微调后的模型重新评估测试集")
    print("2. 对比微调前后的 CER 变化")


if __name__ == "__main__":
    main()
