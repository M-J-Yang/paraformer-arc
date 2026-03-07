"""
实时流式识别模块
使用 FunASR Paraformer Streaming 进行实时语音识别
"""
from typing import Optional, Dict, Any, Callable
from loguru import logger
import torch

try:
    from funasr import AutoModel
except ImportError:
    logger.error("FunASR 未安装，请运行: pip install funasr")
    raise

from utils.config import Config
from atc_grammar import ATCGrammarCorrector
from callsign_fix import CallsignFixer


class StreamingASR:
    """实时流式 ASR"""

    def __init__(
        self,
        device: str = "cuda:0",
        chunk_size: int = 60,  # ms
        hotwords_path: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None,
    ):
        """
        初始化流式 ASR

        Args:
            device: 设备 (cuda:0 或 cpu)
            chunk_size: 音频块大小（毫秒）
            hotwords_path: 热词文件路径
            callback: 识别结果回调函数
        """
        self.config = Config()
        self.device = device
        self.chunk_size = chunk_size
        self.callback = callback
        self.hotwords_path = hotwords_path or str(self.config.HOTWORDS_PATH)

        # 检查 CUDA
        if "cuda" in device and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，切换到 CPU")
            self.device = "cpu"

        logger.info(f"初始化流式 ASR 模型")
        logger.info(f"设备: {self.device}, 块大小: {chunk_size}ms")

        try:
            # 加载流式模型
            self.model = AutoModel(
                model=self.config.STREAMING_MODEL_CONFIG["model_name"],
                device=self.device,
            )
            logger.info("流式 ASR 模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

        # 初始化后处理模块
        self.grammar_corrector = ATCGrammarCorrector()
        self.callsign_fixer = CallsignFixer()

        # 缓存
        self.cache = {}

    def process_chunk(
        self,
        audio_chunk: bytes,
        is_final: bool = False,
    ) -> Optional[str]:
        """
        处理音频块

        Args:
            audio_chunk: 音频数据块
            is_final: 是否为最后一块

        Returns:
            识别文本（如果有）
        """
        try:
            # 调用模型
            result = self.model.generate(
                input=audio_chunk,
                cache=self.cache,
                is_final=is_final,
            )

            # 提取文本
            if isinstance(result, dict) and "text" in result:
                text = result["text"]
            else:
                text = str(result)

            if text:
                logger.debug(f"识别结果: {text}")

                # 后处理
                normalized_text = self.grammar_corrector.normalize(text)
                fixed_text = self.callsign_fixer.fix_callsign(normalized_text)

                # 回调
                if self.callback:
                    self.callback(fixed_text)

                return fixed_text

            return None

        except Exception as e:
            logger.error(f"处理音频块失败: {e}")
            return None

    def reset(self):
        """重置缓存"""
        self.cache = {}
        logger.debug("缓存已重置")

    def stream_from_microphone(self, duration: int = 60):
        """
        从麦克风实时识别

        Args:
            duration: 录音时长（秒）
        """
        try:
            import pyaudio
        except ImportError:
            logger.error("pyaudio 未安装，请运行: pip install pyaudio")
            raise

        logger.info(f"开始从麦克风录音，时长: {duration}秒")

        # 音频参数
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        # 初始化 PyAudio
        p = pyaudio.PyAudio()

        try:
            # 打开音频流
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )

            logger.info("开始录音...")

            frames = []
            for i in range(0, int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK)
                frames.append(data)

                # 每隔一定时间处理一次
                if len(frames) >= 10:  # 约 0.64 秒
                    audio_data = b''.join(frames)
                    self.process_chunk(audio_data, is_final=False)
                    frames = []

            # 处理剩余数据
            if frames:
                audio_data = b''.join(frames)
                self.process_chunk(audio_data, is_final=True)

            logger.info("录音结束")

        finally:
            # 关闭流
            stream.stop_stream()
            stream.close()
            p.terminate()

    def stream_from_file(self, audio_file: str):
        """
        从音频文件模拟流式识别

        Args:
            audio_file: 音频文件路径
        """
        try:
            import soundfile as sf
        except ImportError:
            logger.error("soundfile 未安装，请运行: pip install soundfile")
            raise

        logger.info(f"从文件流式识别: {audio_file}")

        # 读取音频
        audio_data, sample_rate = sf.read(audio_file)

        # 重采样到 16kHz（如果需要）
        if sample_rate != 16000:
            try:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            except ImportError:
                logger.warning("librosa 未安装，无法重采样")

        # 分块处理
        chunk_samples = int(16000 * self.chunk_size / 1000)  # 每块的采样点数

        for i in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[i:i + chunk_samples]
            is_final = (i + chunk_samples >= len(audio_data))

            # 转换为字节
            chunk_bytes = (chunk * 32767).astype('int16').tobytes()

            self.process_chunk(chunk_bytes, is_final=is_final)

        logger.info("流式识别完成")


def main():
    """测试函数"""

    def print_result(text: str):
        """打印识别结果"""
        print(f"[实时识别] {text}")

    # 初始化流式 ASR
    streaming_asr = StreamingASR(
        device="cuda:0",
        callback=print_result
    )

    # 测试：从文件流式识别
    config = Config()
    audio_dir = config.AUDIO_DIR
    test_audio = list(audio_dir.glob("**/*.wav"))

    if test_audio:
        logger.info(f"测试文件: {test_audio[0]}")
        streaming_asr.stream_from_file(str(test_audio[0]))
    else:
        logger.warning("未找到测试音频文件")

    # 测试：从麦克风（需要麦克风设备）
    # streaming_asr.stream_from_microphone(duration=10)


if __name__ == "__main__":
    main()
