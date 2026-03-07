"""
ATC 语法修复模块
处理中文数字转换、高度/航向/频率标准化、ICAO 标准句式修复
"""
import re
from typing import Dict, Tuple, Optional
from loguru import logger
import cn2an


class ATCGrammarCorrector:
    """ATC 语法修复器"""

    def __init__(self):
        # 中文数字映射（航空专用读法）
        self.aviation_digit_map = {
            '洞': '0', '幺': '1', '两': '2', '拐': '7',
            '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'
        }

        # 高度单位
        self.altitude_units = ['米', '英尺', '尺']

        # 方向词
        self.direction_words = ['左转', '右转', '左', '右']

    def convert_aviation_number(self, text: str) -> str:
        """
        转换航空数字读法
        例如: "幺二三" -> "123", "洞洞五" -> "005"
        """
        result = text
        for cn_digit, ar_digit in self.aviation_digit_map.items():
            result = result.replace(cn_digit, ar_digit)
        return result

    def convert_chinese_number(self, text: str) -> str:
        """
        转换中文数字为阿拉伯数字
        例如: "三千" -> "3000", "一万" -> "10000"
        """
        # 常见高度表达
        altitude_patterns = {
            r'一千': '1000',
            r'两千': '2000',
            r'三千': '3000',
            r'四千': '4000',
            r'五千': '5000',
            r'六千': '6000',
            r'七千': '7000',
            r'八千': '8000',
            r'九千': '9000',
            r'一万': '10000',
            r'两万': '20000',
            r'三万': '30000',
        }

        result = text
        for pattern, replacement in altitude_patterns.items():
            result = re.sub(pattern, replacement, result)

        # 处理航向数字 (如 "一八零" -> "180")
        # 匹配连续的中文数字
        def replace_heading(match):
            heading_str = match.group(0)
            digits = ''
            for char in heading_str:
                if char in self.aviation_digit_map:
                    digits += self.aviation_digit_map[char]
                elif char in '0123456789':
                    digits += char
            return digits

        # 匹配航向模式: 数字+数字+数字 (如 "一八零", "三六零")
        result = re.sub(r'[一二三四五六七八九零洞幺]{2,3}(?=度|°|$|\s)', replace_heading, result)

        return result

    def normalize_altitude(self, text: str) -> str:
        """
        标准化高度表达
        例如: "下降到三千" -> "下降至高度3000"
        """
        # 模式: 动作 + 数字 + 可选单位
        patterns = [
            (r'(爬升|上升|上升到|爬升到)(\d+)', r'爬升至高度\2'),
            (r'(下降|下降到)(\d+)', r'下降至高度\2'),
            (r'(保持|保持高度)(\d+)', r'保持高度\2'),
        ]

        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)

        # 移除高度单位词
        for unit in self.altitude_units:
            result = result.replace(unit, '')

        return result

    def normalize_heading(self, text: str) -> str:
        """
        标准化航向表达
        例如: "右转航向一八零" -> "右转航向180"
        """
        # 模式: 方向 + 航向 + 数字
        patterns = [
            (r'(左转|右转)航向(\d+)', r'\1航向\2'),
            (r'(左转|右转)(\d+)', r'\1航向\2'),
            (r'航向(\d+)', r'航向\1'),
        ]

        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)

        return result

    def normalize_frequency(self, text: str) -> str:
        """
        标准化频率表达
        例如: "联系一二三点五" -> "联系频率123.5"
        """
        # 处理频率中的小数点
        # 模式: 数字 + "点" + 数字
        def replace_frequency(match):
            freq_str = match.group(0)
            # 转换 "一二三点五" -> "123.5"
            parts = freq_str.split('点')
            if len(parts) == 2:
                integer_part = self.convert_aviation_number(parts[0])
                decimal_part = self.convert_aviation_number(parts[1])
                return f"{integer_part}.{decimal_part}"
            return freq_str

        result = re.sub(r'[一二三四五六七八九零洞幺\d]+点[一二三四五六七八九零洞幺\d]+', replace_frequency, text)

        # 标准化频率表达
        patterns = [
            (r'(联系|切换|切台)(\d+\.?\d*)', r'\1频率\2'),
            (r'频率(\d+\.?\d*)', r'频率\1'),
        ]

        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)

        return result

    def normalize_runway(self, text: str) -> str:
        """
        标准化跑道表达
        例如: "01号跑道" -> "跑道01"
        """
        # 模式: 数字 + "号" + "跑道"
        result = re.sub(r'(\d+)号跑道', r'跑道\1', text)
        result = re.sub(r'(\d+)跑道', r'跑道\1', result)

        return result

    def add_spacing(self, text: str) -> str:
        """
        在关键词之间添加空格，提高可读性
        """
        # 在数字和中文之间添加空格
        result = re.sub(r'(\d+)([一-龥])', r'\1 \2', text)
        result = re.sub(r'([一-龥])(\d+)', r'\1 \2', result)

        return result

    def normalize(self, text: str) -> str:
        """
        完整的语法标准化流程
        """
        logger.debug(f"原始文本: {text}")

        # 1. 转换航空数字读法
        result = self.convert_aviation_number(text)
        logger.debug(f"航空数字转换: {result}")

        # 2. 转换中文数字
        result = self.convert_chinese_number(result)
        logger.debug(f"中文数字转换: {result}")

        # 3. 标准化高度
        result = self.normalize_altitude(result)
        logger.debug(f"高度标准化: {result}")

        # 4. 标准化航向
        result = self.normalize_heading(result)
        logger.debug(f"航向标准化: {result}")

        # 5. 标准化频率
        result = self.normalize_frequency(result)
        logger.debug(f"频率标准化: {result}")

        # 6. 标准化跑道
        result = self.normalize_runway(result)
        logger.debug(f"跑道标准化: {result}")

        # 7. 添加空格
        result = self.add_spacing(result)
        logger.debug(f"添加空格: {result}")

        # 8. 清理多余空格
        result = re.sub(r'\s+', ' ', result).strip()

        logger.info(f"标准化完成: {text} -> {result}")
        return result


def main():
    """测试函数"""
    corrector = ATCGrammarCorrector()

    test_cases = [
        "南航123右转航向一八零下降到三千",
        "国航856保持一万联系广州进近",
        "东航5187爬升到五千",
        "01号跑道可以起飞",
        "联系塔台频率一二三点五",
        "幺洞洞号跑道脱离",
    ]

    for text in test_cases:
        normalized = corrector.normalize(text)
        print(f"原文: {text}")
        print(f"修复: {normalized}")
        print("-" * 50)


if __name__ == "__main__":
    main()
