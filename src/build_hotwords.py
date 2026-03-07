"""
热词构建模块
从 wordlist.txt 提取 ATC 热词并生成 FunASR 兼容的热词文件
"""
import json
from pathlib import Path
from typing import List, Dict, Set
from loguru import logger

from utils.config import Config


class HotwordBuilder:
    """ATC 热词构建器"""

    def __init__(self):
        self.config = Config()
        self.wordlist_path = self.config.WORDLIST_PATH
        self.hotwords_path = self.config.HOTWORDS_PATH
        self.airline_mapping_path = self.config.AIRLINE_MAPPING_PATH

        # 热词权重配置
        self.high_priority_weight = self.config.HOTWORD_CONFIG["high_priority_weight"]
        self.medium_priority_weight = self.config.HOTWORD_CONFIG["medium_priority_weight"]
        self.low_priority_weight = self.config.HOTWORD_CONFIG["low_priority_weight"]

    def load_wordlist(self) -> List[str]:
        """加载词表文件"""
        words = []
        with open(self.wordlist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '→' in line:
                    # 格式: "编号→词"
                    word = line.split('→')[1]
                    words.append(word)
        logger.info(f"加载词表: {len(words)} 个词")
        return words

    def categorize_hotwords(self, words: List[str]) -> Dict[str, Set[str]]:
        """将词汇分类为不同优先级"""
        categories = {
            "high_priority": set(),
            "medium_priority": set(),
            "low_priority": set(),
        }

        # 高优先级: 航空公司、关键指令
        high_priority_keywords = [
            "起飞", "着陆", "复飞", "进近", "脱离",
            "跑道", "联系", "频率", "高度", "航向"
        ]
        # 添加航空公司
        high_priority_keywords.extend(self.config.AIRLINES)

        # 中优先级: 方向指令、设施
        medium_priority_keywords = [
            "左转", "右转", "保持", "爬升", "下降",
            "滑行", "滑出", "停机位", "停机坪", "塔台",
            "加速", "减速", "等待", "盘旋", "转弯"
        ]

        # 低优先级: 数字表达、方位
        low_priority_keywords = [
            "一千", "三千", "五千", "一万",
            "一边", "两边", "三边", "四边", "五边",
            "东", "南", "西", "北", "左", "右",
            "上升", "下降", "前", "后"
        ]

        for word in words:
            if word in high_priority_keywords or any(airline in word for airline in self.config.AIRLINES):
                categories["high_priority"].add(word)
            elif word in medium_priority_keywords:
                categories["medium_priority"].add(word)
            elif word in low_priority_keywords:
                categories["low_priority"].add(word)
            elif len(word) >= 2:  # 其他多字词作为低优先级
                categories["low_priority"].add(word)

        logger.info(f"高优先级热词: {len(categories['high_priority'])} 个")
        logger.info(f"中优先级热词: {len(categories['medium_priority'])} 个")
        logger.info(f"低优先级热词: {len(categories['low_priority'])} 个")

        return categories

    def build_hotwords_file(self):
        """构建热词文件"""
        # 加载词表
        words = self.load_wordlist()

        # 分类热词
        categories = self.categorize_hotwords(words)

        # 生成热词文件
        hotwords_lines = []

        # 高优先级
        for word in sorted(categories["high_priority"]):
            hotwords_lines.append(f"{word} {self.high_priority_weight}")

        # 中优先级
        for word in sorted(categories["medium_priority"]):
            hotwords_lines.append(f"{word} {self.medium_priority_weight}")

        # 低优先级
        for word in sorted(categories["low_priority"]):
            hotwords_lines.append(f"{word} {self.low_priority_weight}")

        # 写入文件
        self.config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.hotwords_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(hotwords_lines))

        logger.info(f"热词文件已生成: {self.hotwords_path}")
        logger.info(f"总计 {len(hotwords_lines)} 个热词")

    def build_airline_mapping(self):
        """构建航空公司名称映射表"""
        airline_mapping = {
            # 常见误识别
            "男航": "南航",
            "南行": "南航",
            "东行": "东航",
            "过航": "国航",
            "国行": "国航",
            "海行": "海航",
            "川行": "川航",
            "下门": "厦航",
            "厦门": "厦航",
            "深行": "深航",
            "山行": "山航",
            "上行": "上航",
            "春秋行": "春秋",
            "吉祥行": "吉祥",
            "首都行": "首都",
            "天津行": "天津",
            "河北行": "河北",
            "长龙行": "长龙",
        }

        # 保存为 JSON
        self.config.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.airline_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(airline_mapping, f, ensure_ascii=False, indent=2)

        logger.info(f"航空公司映射表已生成: {self.airline_mapping_path}")
        logger.info(f"总计 {len(airline_mapping)} 个映射规则")

    def build_all(self):
        """构建所有热词相关文件"""
        logger.info("开始构建热词文件...")
        self.build_hotwords_file()
        self.build_airline_mapping()
        logger.info("热词构建完成!")


def main():
    """主函数"""
    builder = HotwordBuilder()
    builder.build_all()


if __name__ == "__main__":
    main()
