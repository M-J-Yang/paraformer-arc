# FunASR Paraformer 微调指南

## 当前问题

预训练模型在 ATC 数据集上的表现：
- **测试集 CER: 47.98%**
- **识别准确率: 52.02%**

效果较差的原因：
1. 预训练模型是通用中文 ASR，未针对 ATC 场景优化
2. ATC 语音包含大量专业术语、航空呼号、数字读法
3. 强噪声环境（塔台、无线电）

## 微调方案

### 方案 1: 使用准备好的脚本（推荐）

```bash
# 1. 准备微调数据和配置
cd src
python prepare_finetune.py

# 2. 运行训练脚本
cd ../finetune
train.bat  # Windows
# 或
bash train.sh  # Linux
```

### 方案 2: 使用 FunASR 命令行

```bash
# 安装 FunASR
pip install funasr

# 运行微调
python -m funasr.bin.train \
    --config finetune/output/config.yaml \
    --output_dir finetune/output \
    --ngpu 1
```

### 方案 3: 使用 ModelScope Trainer

```bash
cd src
python finetune_paraformer.py
```

## 微调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 4 | 批量大小（根据 GPU 显存调整） |
| num_epochs | 20 | 训练轮数 |
| learning_rate | 5e-5 | 学习率 |
| warmup_steps | 1000 | 预热步数 |
| grad_clip | 5.0 | 梯度裁剪 |

## 数据集划分

- **训练集**: 80% (用于模型训练)
- **验证集**: 10% (用于超参数调优)
- **测试集**: 10% (用于最终评估)

## 预期效果

微调后预期 CER 降低到：
- **目标 CER: < 10%**
- **识别准确率: > 90%**

## 训练监控

训练过程中关注：
1. **训练损失 (train_loss)**: 应持续下降
2. **验证损失 (val_loss)**: 应下降且不过拟合
3. **验证 CER (val_cer)**: 应持续降低

## 微调后评估

```bash
# 使用微调后的模型评估
cd examples
python evaluate_finetuned.py
```

## 常见问题

### 1. GPU 显存不足
- 减小 batch_size (如改为 2 或 1)
- 使用梯度累积 (accum_grad)

### 2. 训练不收敛
- 降低学习率 (如 1e-5)
- 增加 warmup_steps
- 检查数据质量

### 3. 过拟合
- 增加数据增强
- 使用 dropout
- 早停 (early stopping)

## 目录结构

```
finetune/
├── data/
│   ├── train.txt          # 训练数据列表
│   └── val.txt            # 验证数据列表
├── output/
│   ├── config.yaml        # 训练配置
│   ├── checkpoint/        # 模型检查点
│   └── logs/              # 训练日志
├── train.sh               # Linux 训练脚本
└── train.bat              # Windows 训练脚本
```

## 参考资料

- [FunASR 官方文档](https://github.com/alibaba-damo-academy/FunASR)
- [ModelScope 文档](https://modelscope.cn/)
- [Paraformer 论文](https://arxiv.org/abs/2206.08317)
