"""
呼号识别与修复模块
识别并修复航班呼号（Callsign）
"""
import re
import json
from typing import Optional, Tuple, List
from pathlib import Path
from loguru import logger

from utils.config import Config


class CallsignFixer:
    """航班呼号修复器"""

    def __init__(self):
        self.config = Config()
        self.airline_mapping = self._load_airline_mapping()
        self.airlines = self.config.AIRLINES

    def _load_airline_mapping(self) -> dict:
        """加载航空公司名称映射表"""
        mapping_path = self.config.AIRLINE_MAPPING_PATH
        if mapping_path.exists():
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            logger.info(f"加载航空公司映射表: {len(mapping)} 条规则")
            return mapping
        else:
            logger.warning(f"映射表不存在: {mapping_path}")
            return {}

    def extract_callsign(self, text: str) -> List[Tuple[str, str, str]]:
        """
        提取文本中的呼号
        返回: [(原始呼号, 航空公司, 航班号), ...]
        """
        callsigns = []

        # 模式1: 航空公司名 + 数字 (1-4位)
        # 例如: 南航123, 国航856, 东航5187
        pattern1 = r'([一-龥]{2,4})(\d{1,4})'

        matches = re.finditer(pattern1, text)
        for match in matches:
            airline = match.group(1)
            flight_number = match.group(2)
            original = match.group(0)

            # 检查是否为航空公司名
            if airline in self.airlines or airline in self.airline_mapping:
                callsigns.append((original, airline, flight_number))

        return callsigns

    def fix_airline_name(self, airline: str) -> str:
        """
        修复航空公司名称
        例如: "男航" -> "南航", "东行" -> "东航"
        """
        # 直接映射
        if airline in self.airline_mapping:
            fixed = self.airline_mapping[airline]
            logger.debug(f"修复航空公司名: {airline} -> {fixed}")
            return fixed

        # 已经是正确的
        if airline in self.airlines:
            return airline

        # 模糊匹配（基于编辑距离或拼音相似度）
        # 这里简化处理，只做直接映射
        logger.warning(f"未知航空公司名: {airline}")
        return airline

    def validate_callsign(self, airline: str, flight_number: str) -> bool:
        """
        验证呼号格式是否合法
        """
        # 航空公司名应该是2-4个汉字
        if not (2 <= len(airline) <= 4):
            return False

        # 航班号应该是1-4位数字
        if not (1 <= len(flight_number) <= 4):
            return False

        if not flight_number.isdigit():
            return False

        return True

    def fix_callsign(self, text: str) -> str:
        """
        修复文本中的所有呼号
        """
        result = text
        callsigns = self.extract_callsign(text)

        if not callsigns:
            logger.debug("未检测到呼号")
            return result

        # 按位置倒序处理，避免替换后位置偏移
        callsigns_with_pos = []
        for original, airline, flight_number in callsigns:
            pos = result.find(original)
            if pos != -1:
                callsigns_with_pos.append((pos, original, airline, flight_number))

        callsigns_with_pos.sort(reverse=True)

        for pos, original, airline, flight_number in callsigns_with_pos:
            # 修复航空公司名
            fixed_airline = self.fix_airline_name(airline)

            # 验证呼号
            if self.validate_callsign(fixed_airline, flight_number):
                fixed_callsign = f"{fixed_airline}{flight_number}"

                # 替换
                result = result[:pos] + fixed_callsign + result[pos + len(original):]

                logger.info(f"修复呼号: {original} -> {fixed_callsign}")
            else:
                logger.warning(f"呼号格式不合法: {original}")

        return result

    def extract_all_callsigns(self, text: str) -> List[str]:
        """
        提取文本中所有修复后的呼号
        """
        fixed_text = self.fix_callsign(text)
        callsigns = self.extract_callsign(fixed_text)

        result = []
        for original, airline, flight_number in callsigns:
            fixed_airline = self.fix_airline_name(airline)
            if self.validate_callsign(fixed_airline, flight_number):
                result.append(f"{fixed_airline}{flight_number}")

        return result


def main():
    """测试函数"""
    fixer = CallsignFixer()

    test_cases = [
        "男航123右转航向180下降至高度3000",
        "东行5187保持高度10000",
        "过航856联系广州进近",
        "南航123可以起飞",
        "国航856跑道01脱离",
    ]

    for text in test_cases:
        print(f"\n原始: {text}")
        fixed = fixer.fix_callsign(text)
        print(f"修复: {fixed}")

        callsigns = fixer.extract_all_callsigns(text)
        print(f"呼号: {callsigns}")


if __name__ == "__main__":
    main()
