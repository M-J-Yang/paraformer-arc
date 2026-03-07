"""
实时识别演示
演示如何使用流式 ASR 进行实时识别
"""
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from streaming_asr import StreamingASR
from utils.config import Config
from loguru import logger


def result_callback(text: str):
    """
    识别结果回调函数
    """
    print(f"\n[实时识别] {text}")


def demo_file_streaming():
    """演示：从文件流式识别"""
    logger.info("=== 文件流式识别演示 ===")

    # 初始化流式 ASR
    streaming_asr = StreamingASR(
        device="cuda:0",
        chunk_size=60,
        callback=result_callback
    )

    # 获取测试音频
    config = Config()
    audio_dir = config.AUDIO_DIR
    test_audio = list(audio_dir.glob("**/*.wav"))

    if test_audio:
        audio_file = test_audio[0]
        logger.info(f"测试音频: {audio_file}")

        # 流式识别
        streaming_asr.stream_from_file(str(audio_file))
    else:
        logger.warning("未找到测试音频")


def demo_microphone_streaming():
    """演示：从麦克风实时识别"""
    logger.info("=== 麦克风实时识别演示 ===")

    # 初始化流式 ASR
    streaming_asr = StreamingASR(
        device="cuda:0",
        chunk_size=60,
        callback=result_callback
    )

    try:
        # 从麦克风录音并识别（10 秒）
        streaming_asr.stream_from_microphone(duration=10)
    except Exception as e:
        logger.error(f"麦克风识别失败: {e}")
        logger.info("提示: 需要安装 pyaudio (pip install pyaudio)")


def main():
    """主函数"""
    print("\n=== ATC 实时识别演示 ===\n")
    print("1. 文件流式识别")
    print("2. 麦克风实时识别")
    print()

    choice = input("请选择演示模式 (1/2): ").strip()

    if choice == "1":
        demo_file_streaming()
    elif choice == "2":
        demo_microphone_streaming()
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
