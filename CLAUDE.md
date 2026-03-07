# 中文 ATC 语音识别系统 - 项目规划

## 项目概述

构建工业级中文空管（Air Traffic Control）语音识别系统，用于塔台/空管指挥语音的自动识别与结构化输出。

## 系统架构

```
Audio Input (WAV/PCM)
    ↓
Audio Preprocessing (重采样 16kHz + VAD + 降噪)
    ↓
ASR Model (FunASR Paraformer-zh + VAD + Punctuation)
    ↓
Hotword Boosting (ATC 热词增强)
    ↓
ATC Grammar Correction (ICAO 标准句式修复)
    ↓
Callsign Detection & Repair (航班号识别修复)
    ↓
Structured Output (JSON 格式化 ATC 指令)
```

## 数据资源

### 现有数据集
- **位置**: `D:\NPU_works\语音\ATC-paraformer\chinese_ATC_formatted`
- **音频数据**: `WAVdata/` - 中文 ATC 语音样本
- **文本数据**: `TXTdata/` - 对应转录文本
- **词表**: `TXTdata/wordlist.txt` - 921 个 ATC 专业词汇

### 词表分析
词表包含：
- 基础字符：数字 0-9、字母 A-Z/a-z
- 单字词：一、二、三、上、下、左、右、高、低等
- ATC 术语：跑道、进近、复飞、起飞、着陆、滑行、爬升、下降等
- 复合词：三转弯、上升气流、下沉气流、停机位、跑道入侵等
- 航空公司：锦州、长江等（需扩展：国航、南航、东航等）
- 气象术语：风切变、雷暴、颠簸、能见度等

## 核心模块设计

### 1. ASR 推理模块 (`asr_infer.py`)
**功能**:
- 加载 FunASR Paraformer-zh 模型
- 支持 VAD（语音活动检测）
- 支持标点预测
- GPU 加速推理
- 批量推理支持

**技术栈**:
```python
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",           # 或 paraformer-zh-streaming
    vad_model="fsmn-vad",            # VAD 模型
    punc_model="ct-punc",            # 标点模型
    device="cuda:0",                 # GPU 推理
    ncpu=4                           # CPU 线程数
)
```

**输入**: 音频文件路径或音频数据
**输出**: 原始 ASR 文本

---

### 2. 热词构建模块 (`build_hotwords.py`)
**功能**:
- 从 wordlist.txt 提取 ATC 热词
- 构建航空公司呼号词表
- 生成 FunASR 兼容的热词文件
- 支持热词权重配置

**热词分类**:
1. **高优先级热词** (权重 20-30):
   - 航空公司：国航、南航、东航、海航、川航、厦航、深航
   - 关键指令：起飞、着陆、复飞、进近、脱离

2. **中优先级热词** (权重 10-20):
   - 方向指令：左转、右转、保持、爬升、下降
   - 设施：跑道、滑行道、停机位、塔台

3. **低优先级热词** (权重 5-10):
   - 数字表达：一千、三千、五千、一万
   - 方位：东、南、西、北、左、右

**输出格式** (hotwords.txt):
```
国航 30
南航 30
东航 30
起飞 25
着陆 25
跑道 20
...
```

---

### 3. ATC 语法修复模块 (`atc_grammar.py`)
**功能**:
- 中文数字转阿拉伯数字
- 高度/航向/频率标准化
- ICAO 标准句式修复
- 单位统一

**转换规则**:

#### 数字转换
```python
# 中文数字 → 阿拉伯数字
"三千" → "3000"
"一万" → "10000"
"一八零" → "180"
"幺二三" → "123"
"洞洞五" → "005"
```

#### 高度标准化
```python
# 统一高度表达
"三千米" → "高度3000"
"保持一万" → "保持高度10000"
"下降到五千" → "下降至高度5000"
```

#### 航向标准化
```python
# 统一航向表达
"右转航向一八零" → "右转航向180"
"左转航向三六零" → "左转航向360"
```

#### 频率标准化
```python
# 统一频率表达
"联系一二三点五" → "联系频率123.5"
"切换到一一八点一" → "切换频率118.1"
```

**输入**: 原始 ASR 文本
**输出**: 标准化 ATC 文本

---

### 4. 呼号修复模块 (`callsign_fix.py`)
**功能**:
- 识别航班呼号结构
- 修复 ASR 误识别的航空公司名
- 验证呼号格式

**修复规则**:
```python
# 航空公司名称映射
AIRLINE_MAPPING = {
    "男航": "南航",
    "南行": "南航",
    "东行": "东航",
    "过航": "国航",
    "国行": "国航",
    "海行": "海航",
    "川行": "川航",
    "下门": "厦航",
    "深行": "深航",
}

# 呼号格式: 航空公司 + 数字
# 例如: 南航123, 国航856, 东航5187
```

**识别流程**:
1. 使用正则提取潜在呼号（航空公司名 + 数字）
2. 应用航空公司名称映射
3. 验证数字部分（1-4 位数字）
4. 返回标准化呼号

**输入**: 标准化 ATC 文本
**输出**: 呼号修复后的文本

---

### 5. 实时识别模块 (`streaming_asr.py`)
**功能**:
- 流式音频输入处理
- 实时 ASR 识别
- 低延迟输出（< 500ms）

**技术方案**:
```python
from funasr import AutoModel

# 使用流式模型
model = AutoModel(
    model="paraformer-zh-streaming",
    device="cuda:0"
)

# 流式识别
for chunk in audio_stream:
    result = model.generate(
        input=chunk,
        cache={},  # 保持上下文
        is_final=False
    )
```

**输入**: 音频流（实时麦克风/网络流）
**输出**: 实时 ASR 结果

---

### 6. 主 Pipeline (`atc_asr_pipeline.py`)
**功能**:
- 整合所有模块
- 端到端处理
- 结构化 JSON 输出

**处理流程**:
```python
def process_atc_audio(audio_path):
    # 1. ASR 识别
    raw_text = asr_model.transcribe(audio_path, hotwords=hotwords)

    # 2. 语法修复
    normalized_text = grammar_corrector.normalize(raw_text)

    # 3. 呼号修复
    fixed_text = callsign_fixer.fix(normalized_text)

    # 4. 结构化输出
    structured_output = parse_atc_command(fixed_text)

    return structured_output
```

**输出格式**:
```json
{
  "raw_text": "南航123右转航向一八零下降到三千",
  "normalized_text": "南航123 右转航向180 下降至高度3000",
  "callsign": "南航123",
  "commands": [
    {
      "type": "heading_change",
      "action": "右转",
      "heading": 180
    },
    {
      "type": "altitude_change",
      "action": "下降",
      "altitude": 3000
    }
  ],
  "timestamp": "2026-03-06T12:00:00Z"
}
```

---

## 指令类型分类

### 1. 航向指令 (Heading)
- 关键词: 右转、左转、航向、保持航向
- 参数: 航向角度 (0-360)
- 示例: "右转航向180"

### 2. 高度指令 (Altitude)
- 关键词: 爬升、下降、保持高度
- 参数: 高度值（米/英尺）
- 示例: "下降至高度3000"

### 3. 速度指令 (Speed)
- 关键词: 加速、减速、保持速度
- 参数: 速度值（节/公里/小时）
- 示例: "保持速度250"

### 4. 起降指令 (Takeoff/Landing)
- 关键词: 起飞、着陆、复飞、进近
- 参数: 跑道号
- 示例: "01号跑道可以起飞"

### 5. 滑行指令 (Taxi)
- 关键词: 滑行、滑出、脱离、停机位
- 参数: 滑行道、停机位号
- 示例: "滑行至A5停机位"

### 6. 频率切换 (Frequency)
- 关键词: 联系、切换、频率
- 参数: 频率值、管制单位
- 示例: "联系广州进近频率123.5"

### 7. 等待指令 (Hold)
- 关键词: 等待、盘旋、保持
- 参数: 等待点、高度
- 示例: "在XX点等待高度3000"

---

## 项目文件结构

```
D:\NPU_works\语音\ATC-paraformer\
├── CLAUDE.md                    # 本文档
├── README.md                    # 项目说明
├── requirements.txt             # Python 依赖
├── chinese_ATC_formatted/       # 数据集
│   ├── WAVdata/                # 音频文件
│   └── TXTdata/                # 文本标注
│       └── wordlist.txt        # 词表
├── src/                        # 源代码
│   ├── __init__.py
│   ├── asr_infer.py           # ASR 推理
│   ├── build_hotwords.py      # 热词构建
│   ├── atc_grammar.py         # 语法修复
│   ├── callsign_fix.py        # 呼号修复
│   ├── streaming_asr.py       # 实时识别
│   ├── atc_asr_pipeline.py    # 主 Pipeline
│   └── utils/                 # 工具函数
│       ├── audio_utils.py     # 音频处理
│       ├── text_utils.py      # 文本处理
│       └── config.py          # 配置管理
├── config/                     # 配置文件
│   ├── model_config.yaml      # 模型配置
│   ├── hotwords.txt           # 热词表
│   └── airline_mapping.json   # 航空公司映射
├── tests/                      # 测试代码
│   ├── test_asr.py
│   ├── test_grammar.py
│   └── test_pipeline.py
└── examples/                   # 示例代码
    ├── batch_inference.py     # 批量推理
    └── realtime_demo.py       # 实时演示
```

---

## 开发计划

### Phase 1: 基础设施 (第1-2天)
- [x] 项目结构搭建
- [ ] 环境配置 (FunASR, PyTorch, CUDA)
- [ ] 数据集探索与分析
- [ ] 热词表构建

### Phase 2: 核心模块 (第3-5天)
- [ ] ASR 推理模块开发
- [ ] 语法修复模块开发
- [ ] 呼号修复模块开发
- [ ] 单元测试

### Phase 3: Pipeline 集成 (第6-7天)
- [ ] 主 Pipeline 开发
- [ ] 结构化输出设计
- [ ] 批量推理脚本
- [ ] 性能优化

### Phase 4: 实时系统 (第8-9天)
- [ ] 流式识别模块
- [ ] 实时 Pipeline
- [ ] 延迟优化
- [ ] 实时演示

### Phase 5: 测试与优化 (第10天)
- [ ] 端到端测试
- [ ] 准确率评估
- [ ] 性能调优
- [ ] 文档完善

---

## 技术要求

### 环境依赖
```
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.8
FunASR >= 1.0
librosa >= 0.10
soundfile >= 0.12
numpy >= 1.24
```

### 硬件要求
- GPU: NVIDIA GPU with >= 8GB VRAM (推荐 RTX 3090/4090)
- RAM: >= 16GB
- Storage: >= 50GB (模型 + 数据集)

### 性能指标
- **准确率**: WER < 10% (Word Error Rate)
- **延迟**: < 500ms (实时系统)
- **吞吐量**: >= 10x 实时 (批量推理)

---

## 下一步行动

1. **创建项目结构**: 建立 `src/`, `config/`, `tests/` 目录
2. **安装依赖**: 配置 FunASR 环境
3. **构建热词表**: 从 wordlist.txt 提取并扩展热词
4. **开发 ASR 模块**: 实现基础推理功能
5. **逐步集成**: 按模块顺序开发并测试

---

## 注意事项

1. **热词权重调优**: 需要在实际数据上测试不同权重的效果
2. **数字识别**: 中文数字识别是难点，需要特别处理"幺"、"洞"等航空用语
3. **呼号识别**: 航空公司名称容易误识别，需要完善映射表
4. **实时性能**: 流式识别需要在准确率和延迟之间平衡
5. **鲁棒性**: 需要处理噪声、口音、语速变化等实际场景问题

---

## 参考资料

- FunASR 官方文档: https://github.com/alibaba-damo-academy/FunASR
- Paraformer 模型: https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
- ICAO 标准陆空通话用语: 国际民航组织附件10
