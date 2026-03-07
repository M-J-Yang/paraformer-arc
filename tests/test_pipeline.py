"""
测试完整 Pipeline
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from atc_asr_pipeline import ATCASRPipeline
from utils.config import Config


class TestPipeline:
    """测试完整 Pipeline"""

    @pytest.fixture
    def pipeline(self):
        """初始化 Pipeline"""
        return ATCASRPipeline(device="cpu")

    @pytest.fixture
    def test_audio(self):
        """获取测试音频"""
        config = Config()
        audio_files = list(config.AUDIO_DIR.glob("**/*.wav"))
        if audio_files:
            return str(audio_files[0])
        return None

    def test_pipeline_initialization(self, pipeline):
        """测试 Pipeline 初始化"""
        assert pipeline.asr is not None
        assert pipeline.grammar_corrector is not None
        assert pipeline.callsign_fixer is not None

    def test_command_type_parsing(self, pipeline):
        """测试指令类型识别"""
        text = "右转航向180下降至高度3000"
        command_types = pipeline.parse_command_type(text)
        assert "heading_change" in command_types
        assert "altitude_change" in command_types

    def test_parameter_extraction(self, pipeline):
        """测试参数提取"""
        text = "右转航向180下降至高度3000联系频率123.5"
        params = pipeline.extract_parameters(text)

        assert "heading" in params
        assert params["heading"] == 180

        assert "altitude" in params
        assert params["altitude"] == 3000

        assert "frequency" in params
        assert params["frequency"] == 123.5

    def test_command_parsing(self, pipeline):
        """测试指令解析"""
        text = "右转航向180下降至高度3000"
        commands = pipeline.parse_commands(text)

        assert len(commands) >= 2
        assert any(cmd["type"] == "heading_change" for cmd in commands)
        assert any(cmd["type"] == "altitude_change" for cmd in commands)

    def test_process_single(self, pipeline, test_audio):
        """测试单文件处理"""
        if test_audio is None:
            pytest.skip("没有测试音频")

        result = pipeline.process(test_audio)

        assert isinstance(result, dict)
        assert "raw_text" in result
        assert "normalized_text" in result
        assert "fixed_text" in result
        assert "callsign" in result
        assert "commands" in result
        assert "timestamp" in result

    def test_batch_process(self, pipeline):
        """测试批量处理"""
        config = Config()
        audio_files = list(config.AUDIO_DIR.glob("**/*.wav"))[:2]

        if not audio_files:
            pytest.skip("没有测试音频")

        results = pipeline.batch_process(audio_files, batch_size=2)

        assert isinstance(results, list)
        assert len(results) == len(audio_files)
        for result in results:
            assert "raw_text" in result
            assert "commands" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
