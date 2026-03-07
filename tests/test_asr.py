"""
测试 ASR 推理模块
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from asr_infer import ASRInference
from utils.config import Config


class TestASRInference:
    """测试 ASR 推理"""

    @pytest.fixture
    def asr(self):
        """初始化 ASR"""
        return ASRInference(device="cpu")  # 使用 CPU 测试

    @pytest.fixture
    def test_audio(self):
        """获取测试音频"""
        config = Config()
        audio_files = list(config.AUDIO_DIR.glob("**/*.wav"))
        if audio_files:
            return str(audio_files[0])
        return None

    def test_model_loading(self, asr):
        """测试模型加载"""
        assert asr.model is not None
        assert asr.device in ["cpu", "cuda:0"]

    def test_transcribe_single(self, asr, test_audio):
        """测试单文件识别"""
        if test_audio is None:
            pytest.skip("没有测试音频")

        result = asr.transcribe(test_audio)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_transcribe_batch(self, asr):
        """测试批量识别"""
        config = Config()
        audio_files = list(config.AUDIO_DIR.glob("**/*.wav"))[:3]

        if not audio_files:
            pytest.skip("没有测试音频")

        results = asr.transcribe(audio_files, batch_size=2)
        assert isinstance(results, list)
        assert len(results) == len(audio_files)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
