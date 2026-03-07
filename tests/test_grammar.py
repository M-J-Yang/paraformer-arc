"""
测试语法修复模块
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from atc_grammar import ATCGrammarCorrector


class TestATCGrammar:
    """测试 ATC 语法修复"""

    @pytest.fixture
    def corrector(self):
        """初始化语法修复器"""
        return ATCGrammarCorrector()

    def test_aviation_number_conversion(self, corrector):
        """测试航空数字转换"""
        assert corrector.convert_aviation_number("幺二三") == "123"
        assert corrector.convert_aviation_number("洞洞五") == "005"
        assert corrector.convert_aviation_number("拐八九") == "789"

    def test_chinese_number_conversion(self, corrector):
        """测试中文数字转换"""
        text = "三千"
        result = corrector.convert_chinese_number(text)
        assert "3000" in result

        text = "一万"
        result = corrector.convert_chinese_number(text)
        assert "10000" in result

    def test_altitude_normalization(self, corrector):
        """测试高度标准化"""
        text = "下降到3000"
        result = corrector.normalize_altitude(text)
        assert "下降至高度3000" in result

        text = "爬升到5000"
        result = corrector.normalize_altitude(text)
        assert "爬升至高度5000" in result

    def test_heading_normalization(self, corrector):
        """测试航向标准化"""
        text = "右转180"
        result = corrector.normalize_heading(text)
        assert "右转航向180" in result

    def test_frequency_normalization(self, corrector):
        """测试频率标准化"""
        text = "联系123.5"
        result = corrector.normalize_frequency(text)
        assert "频率" in result

    def test_full_normalization(self, corrector):
        """测试完整标准化流程"""
        test_cases = [
            ("南航123右转航向一八零下降到三千", "180", "3000"),
            ("国航856保持一万", "10000", None),
            ("东航5187爬升到五千", "5000", None),
        ]

        for input_text, expected_heading, expected_altitude in test_cases:
            result = corrector.normalize(input_text)
            assert isinstance(result, str)
            if expected_heading:
                assert expected_heading in result
            if expected_altitude:
                assert expected_altitude in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
