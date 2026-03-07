"""
ASR 推理模块
使用 FunASR Paraformer 进行语音识别
"""
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from loguru import logger
import torch

try:
    from funasr import AutoModel
except ImportError:
    logger.error("FunASR 未安装，请运行: pip install funasr")
    raise

from utils.config import Config


class ASRInference:
    """ASR 推理器"""

    def __init__(
        self,
        model_name: str = "paraformer-zh",
        vad_model: str = "fsmn-vad",
        punc_model: str = "ct-punc",
        device: str = "cuda:0",
        hotwords_path: Optional[str] = None,
    ):
        """
        初始化 ASR 模型

        Args:
            model_name: 模型名称
            vad_model: VAD 模型名称
            punc_model: 标点模型名称
            device: 设备 (cuda:0 或 cpu)
            hotwords_path: 热词文件路径
        """
        self.config = Config()
        self.device = device
        self.hotwords_path = hotwords_path or str(self.config.HOTWORDS_PATH)

        # 检查 CUDA 可用性
        if "cuda" in device and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，切换到 CPU")
            self.device = "cpu"

        logger.info(f"初始化 ASR 模型: {model_name}")
        logger.info(f"设备: {self.device}")

        try:
            # 加载模型
            self.model = AutoModel(
                model=model_name,
                vad_model=vad_model,
                punc_model=punc_model,
                device=self.device,
                ncpu=self.config.MODEL_CONFIG.get("ncpu", 4),
            )
            logger.info("ASR 模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

        # 加载热词
        self.hotwords = self._load_hotwords()

    def _load_hotwords(self) -> Optional[str]:
        """加载热词文件"""
        if Path(self.hotwords_path).exists():
            logger.info(f"加载热词文件: {self.hotwords_path}")
            return self.hotwords_path
        else:
            logger.warning(f"热词文件不存在: {self.hotwords_path}")
            return None

    def transcribe(
        self,
        audio_input: Union[str, Path, List[str]],
        use_hotwords: bool = True,
        batch_size: int = 1,
    ) -> Union[str, List[str]]:
        """
        语音识别

        Args:
            audio_input: 音频文件路径或路径列表
            use_hotwords: 是否使用热词
            batch_size: 批量大小

        Returns:
            识别文本或文本列表
        """
        # 处理输入
        if isinstance(audio_input, (str, Path)):
            audio_input = [str(audio_input)]
            single_input = True
        else:
            audio_input = [str(p) for p in audio_input]
            single_input = False

        logger.info(f"开始识别 {len(audio_input)} 个音频文件")

        # 准备参数
        kwargs = {}
        if use_hotwords and self.hotwords:
            kwargs["hotword"] = self.hotwords

        try:
            # 批量推理
            results = []
            for i in range(0, len(audio_input), batch_size):
                batch = audio_input[i:i + batch_size]
                logger.debug(f"处理批次 {i // batch_size + 1}: {len(batch)} 个文件")

                # 调用模型
                batch_results = self.model.generate(
                    input=batch,
                    batch_size=len(batch),
                    **kwargs
                )

                # 提取文本
                if isinstance(batch_results, list):
                    for result in batch_results:
                        if isinstance(result, dict) and "text" in result:
                            results.append(result["text"])
                        else:
                            results.append(str(result))
                elif isinstance(batch_results, dict) and "text" in batch_results:
                    results.append(batch_results["text"])
                else:
                    results.append(str(batch_results))

            logger.info(f"识别完成: {len(results)} 个结果")

            # 返回结果
            if single_input:
                return results[0] if results else ""
            else:
                return results

        except Exception as e:
            logger.error(f"识别失败: {e}")
            raise

    def transcribe_with_timestamps(
        self,
        audio_input: Union[str, Path],
        use_hotwords: bool = True,
    ) -> Dict[str, Any]:
        """
        语音识别（带时间戳）

        Args:
            audio_input: 音频文件路径
            use_hotwords: 是否使用热词

        Returns:
            包含文本和时间戳的字典
        """
        logger.info(f"识别音频（带时间戳）: {audio_input}")

        kwargs = {}
        if use_hotwords and self.hotwords:
            kwargs["hotword"] = self.hotwords

        try:
            result = self.model.generate(
                input=str(audio_input),
                batch_size=1,
                **kwargs
            )

            return result

        except Exception as e:
            logger.error(f"识别失败: {e}")
            raise


def main():
    """测试函数"""
    # 初始化 ASR
    asr = ASRInference(device="cuda:0")

    # 测试音频文件
    config = Config()
    audio_dir = config.AUDIO_DIR

    # 查找测试音频
    test_audio = list(audio_dir.glob("**/*.wav"))[:5]

    if test_audio:
        logger.info(f"找到 {len(test_audio)} 个测试音频")

        # 单个识别
        for audio_path in test_audio:
            text = asr.transcribe(audio_path)
            print(f"\n音频: {audio_path.name}")
            print(f"识别: {text}")

        # 批量识别
        texts = asr.transcribe(test_audio, batch_size=2)
        print("\n批量识别结果:")
        for audio_path, text in zip(test_audio, texts):
            print(f"{audio_path.name}: {text}")
    else:
        logger.warning("未找到测试音频文件")


if __name__ == "__main__":
    main()
