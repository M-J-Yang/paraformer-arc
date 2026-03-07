"""
使用 FunASR 官方接口进行微调
基于 FunASR 的标准训练流程
"""
import os
import json
import sys
from pathlib import Path
from typing import Dict, List
import torch
from loguru import logger


def prepare_data_list(samples: List[Dict], output_file: Path):
    """
    准备 FunASR 标准格式的数据列表

    格式: wav_path\ttranscript
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            audio_path = sample["audio"]
            text = sample["gt"]
            f.write(f"{audio_path}\t{text}\n")

    logger.info(f"数据列表已保存: {output_file} ({len(samples)} 样本)")


def create_config_yaml(
    train_data: str,
    val_data: str,
    output_dir: str,
    base_model_dir: str,
    batch_size: int = 4,
    num_epochs: int = 20,
    learning_rate: float = 5e-5,
) -> str:
    """
    创建 FunASR 训练配置文件 (YAML)
    """
    config_content = f"""# FunASR Paraformer 微调配置

# 模型配置
model: paraformer
model_conf:
    ctc_weight: 0.3
    lsm_weight: 0.1
    length_normalized_loss: true

# 预训练模型路径
init_param: {base_model_dir}/model.pt

# 数据配置
dataset: AudioDataset
dataset_conf:
    index_ds: IndexDSJsonl
    batch_sampler: BatchSampler
    batch_type: length
    batch_size: {batch_size}
    max_token_length: 2048
    buffer_size: 500
    shuffle: true
    num_workers: 4

# 训练数据
train_data_path_and_name_and_type:
    - ["{train_data}", "speech", "sound"]

# 验证数据
valid_data_path_and_name_and_type:
    - ["{val_data}", "speech", "sound"]

# 优化器配置
optim: adam
optim_conf:
    lr: {learning_rate}
    weight_decay: 0.000001

# 学习率调度
scheduler: warmuplr
scheduler_conf:
    warmup_steps: 1000

# 训练配置
max_epoch: {num_epochs}
num_att_plot: 0
num_workers: 4
log_interval: 50
keep_nbest_models: 3
grad_clip: 5.0
accum_grad: 1

# 输出目录
output_dir: {output_dir}

# GPU 配置
ngpu: 1
"""

    config_path = Path(output_dir) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

    logger.info(f"配置文件已保存: {config_path}")
    return str(config_path)


def create_train_script(
    config_yaml: str,
    output_dir: str,
    gpu_id: int = 0
) -> str:
    """
    创建训练脚本
    """
    config_dir = Path(config_yaml).parent
    config_name = Path(config_yaml).stem

    script_content = f"""#!/bin/bash
# FunASR Paraformer 微调脚本

export CUDA_VISIBLE_DEVICES={gpu_id}

python -m funasr.bin.train \\
    --config-path {config_dir} \\
    --config-name {config_name} \\
    ++output_dir={output_dir} \\
    ++ngpu=1

echo "训练完成！"
"""

    script_path = Path(output_dir) / "train.sh"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    # Windows 版本
    script_content_win = f"""@echo off
REM FunASR Paraformer 微调脚本 (Windows)

set CUDA_VISIBLE_DEVICES={gpu_id}

python -m funasr.bin.train ^
    --config-path {config_dir} ^
    --config-name {config_name} ^
    ++output_dir={output_dir} ^
    ++ngpu=1

echo 训练完成！
pause
"""

    script_path_win = Path(output_dir) / "train.bat"
    with open(script_path_win, 'w', encoding='utf-8') as f:
        f.write(script_content_win)

    logger.info(f"训练脚本已保存: {script_path} / {script_path_win}")
    return str(script_path_win)


def main():
    """主函数"""
    # 路径配置
    project_root = Path("D:/NPU_works/语音/ATC-paraformer")
    split_dir = project_root / "data_splits"
    finetune_dir = project_root / "finetune"
    finetune_data_dir = finetune_dir / "data"
    finetune_output_dir = finetune_dir / "output"

    # 加载数据集划分
    logger.info("加载数据集划分...")
    with open(split_dir / "train.json", 'r', encoding='utf-8') as f:
        train_samples = json.load(f)
    with open(split_dir / "val.json", 'r', encoding='utf-8') as f:
        val_samples = json.load(f)
    with open(split_dir / "test.json", 'r', encoding='utf-8') as f:
        test_samples = json.load(f)

    total_samples = len(train_samples) + len(val_samples) + len(test_samples)
    logger.info(f"训练集: {len(train_samples)} 样本 ({len(train_samples)/total_samples*100:.1f}%)")
    logger.info(f"验证集: {len(val_samples)} 样本 ({len(val_samples)/total_samples*100:.1f}%)")
    logger.info(f"测试集: {len(test_samples)} 样本 ({len(test_samples)/total_samples*100:.1f}%)")

    # 准备数据列表
    logger.info("\n准备训练数据...")
    finetune_data_dir.mkdir(parents=True, exist_ok=True)

    train_list = finetune_data_dir / "train.txt"
    val_list = finetune_data_dir / "val.txt"
    test_list = finetune_data_dir / "test.txt"

    prepare_data_list(train_samples, train_list)
    prepare_data_list(val_samples, val_list)
    prepare_data_list(test_samples, test_list)

    # 获取预训练模型路径
    cache_dir = Path.home() / ".cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

    if not cache_dir.exists():
        logger.error(f"预训练模型未找到: {cache_dir}")
        logger.info("请先运行一次 ASR 推理以下载模型")
        return

    logger.info(f"预训练模型路径: {cache_dir}")

    # 创建配置文件
    logger.info("\n创建训练配置...")
    config_yaml = create_config_yaml(
        train_data=str(train_list),
        val_data=str(val_list),
        output_dir=str(finetune_output_dir),
        base_model_dir=str(cache_dir),
        batch_size=4,
        num_epochs=20,
        learning_rate=5e-5,
    )

    # 创建训练脚本
    logger.info("\n创建训练脚本...")
    train_script = create_train_script(
        config_yaml=config_yaml,
        output_dir=str(finetune_output_dir),
        gpu_id=0
    )

    # 打印说明
    print("\n" + "=" * 80)
    print("微调准备完成！")
    print("=" * 80)
    print(f"\n数据集划分 (8:1:1):")
    print(f"  训练集: {len(train_samples)} 样本 ({len(train_samples)/total_samples*100:.1f}%)")
    print(f"  验证集: {len(val_samples)} 样本 ({len(val_samples)/total_samples*100:.1f}%)")
    print(f"  测试集: {len(test_samples)} 样本 ({len(test_samples)/total_samples*100:.1f}%)")
    print(f"  总计:   {total_samples} 样本")
    print(f"\n数据准备:")
    print(f"  训练数据: {train_list}")
    print(f"  验证数据: {val_list}")
    print(f"  测试数据: {test_list}")
    print(f"\n配置文件:")
    print(f"  {config_yaml}")
    print(f"\n训练脚本:")
    print(f"  {train_script}")
    print(f"\n开始训练:")
    print(f"  方法1 (推荐): 双击运行 {train_script}")
    print(f"  方法2: cd {finetune_dir} && train.bat")
    print(f"\n训练参数:")
    print(f"  批量大小: 4")
    print(f"  训练轮数: 20")
    print(f"  学习率: 5e-5")
    print(f"  输出目录: {finetune_output_dir}")
    print("=" * 80)

    # 提示手动训练
    logger.info("\n注意: FunASR 微调需要手动运行训练脚本")
    logger.info("如果遇到问题，请参考 FunASR 官方文档:")
    logger.info("https://github.com/alibaba-damo-academy/FunASR")


if __name__ == "__main__":
    main()
