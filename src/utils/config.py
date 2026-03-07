"""
ATC ASR 系统配置管理模块
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml


class Config:
    """配置管理类"""

    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    # 数据目录
    DATA_DIR = PROJECT_ROOT / "chinese_ATC_formatted"
    AUDIO_DIR = DATA_DIR / "WAVdata"
    TEXT_DIR = DATA_DIR / "TXTdata"
    WORDLIST_PATH = TEXT_DIR / "wordlist.txt"

    # 配置目录
    CONFIG_DIR = PROJECT_ROOT / "config"
    HOTWORDS_PATH = CONFIG_DIR / "hotwords.txt"
    AIRLINE_MAPPING_PATH = CONFIG_DIR / "airline_mapping.json"

    # 模型配置
    MODEL_CONFIG = {
        "model_name": "paraformer-zh",
        "vad_model": "fsmn-vad",
        "punc_model": "ct-punc",
        "device": "cuda:0",  # 或 "cpu"
        "ncpu": 4,
        "batch_size": 1,
    }

    # 流式模型配置
    STREAMING_MODEL_CONFIG = {
        "model_name": "paraformer-zh-streaming",
        "device": "cuda:0",
        "chunk_size": 60,  # ms
    }

    # 热词配置
    HOTWORD_CONFIG = {
        "high_priority_weight": 30,  # 航空公司、关键指令
        "medium_priority_weight": 20,  # 方向指令、设施
        "low_priority_weight": 10,  # 数字表达、方位
    }

    # ATC 指令类型
    COMMAND_TYPES = {
        "heading_change": ["右转", "左转", "航向", "转弯"],
        "altitude_change": ["爬升", "下降", "保持高度", "上升", "下降到"],
        "speed_change": ["加速", "减速", "保持速度"],
        "takeoff_landing": ["起飞", "着陆", "复飞", "进近", "落地"],
        "taxi": ["滑行", "滑出", "脱离", "停机位", "停机坪"],
        "frequency_change": ["联系", "切换", "频率", "切台"],
        "hold": ["等待", "盘旋", "保持"],
    }

    # 航空公司列表
    AIRLINES = [
        "国航", "南航", "东航", "海航", "川航",
        "厦航", "深航", "山航", "上航", "春秋",
        "吉祥", "首都", "天津", "河北", "长龙"
    ]

    @classmethod
    def load_yaml(cls, config_path: str) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @classmethod
    def save_yaml(cls, data: Dict[str, Any], config_path: str):
        """保存 YAML 配置文件"""
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
