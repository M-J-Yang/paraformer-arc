"""
直接使用 FunASR API 进行微调
避免 Hydra 配置问题
"""
import os
import sys
from pathlib import Path

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

try:
    from funasr import AutoModel
    from modelscope.trainers import build_trainer
    from modelscope.utils.config import Config
except ImportError as e:
    logger.error(f"导入失败: {e}")
    logger.info("请确保已安装: pip install funasr modelscope")
    sys.exit(1)


def train_with_funasr():
    """使用 FunASR 进行微调"""

    # 配置路径
    project_root = Path(__file__).parent.parent
    finetune_dir = project_root / "finetune"

    train_data = finetune_dir / "data" / "train.txt"
    val_data = finetune_dir / "data" / "val.txt"
    model_dir = finetune_dir / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    output_dir = finetune_dir / "output"

    logger.info("=" * 80)
    logger.info("开始 FunASR Paraformer 微调")
    logger.info("=" * 80)
    logger.info(f"训练数据: {train_data}")
    logger.info(f"验证数据: {val_data}")
    logger.info(f"预训练模型: {model_dir}")
    logger.info(f"输出目录: {output_dir}")

    # 检查文件
    if not train_data.exists():
        logger.error(f"训练数据不存在: {train_data}")
        return
    if not val_data.exists():
        logger.error(f"验证数据不存在: {val_data}")
        return
    if not model_dir.exists():
        logger.error(f"预训练模型不存在: {model_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练配置
    config = {
        'task': 'asr',
        'model': str(model_dir),
        'train_dataset': {
            'type': 'AudioDataset',
            'data_path': str(train_data),
        },
        'eval_dataset': {
            'type': 'AudioDataset',
            'data_path': str(val_data),
        },
        'work_dir': str(output_dir),
        'max_epochs': 20,
        'batch_size': 4,
        'lr': 5e-5,
        'warmup_steps': 1000,
        'grad_clip': 5.0,
        'log_interval': 50,
        'save_checkpoint_interval': 1,
        'keep_checkpoint_max': 3,
    }

    logger.info("\n训练配置:")
    for key, value in config.items():
        if not isinstance(value, dict):
            logger.info(f"  {key}: {value}")

    try:
        logger.info("\n开始训练...")

        # 方法1: 使用 ModelScope Trainer
        trainer = build_trainer(
            name='speech-recognition-trainer',
            default_args=config
        )

        trainer.train()

        logger.info("\n训练完成！")
        logger.info(f"模型已保存到: {output_dir}")

    except Exception as e:
        logger.error(f"\n训练失败: {e}")
        logger.info("\n请尝试以下方法:")
        logger.info("1. 检查 FunASR 版本: pip show funasr")
        logger.info("2. 查看 FunASR 文档: https://github.com/alibaba-damo-academy/FunASR")
        logger.info("3. 使用命令行工具进行微调")


if __name__ == "__main__":
    train_with_funasr()
