"""
将 train.txt/val.txt 转换为 FunASR 需要的 JSONL 格式
"""
import json
import sys
from pathlib import Path
from loguru import logger


def txt_to_jsonl(txt_file: Path, jsonl_file: Path):
    """
    将 wav_path\ttext 格式转换为 JSONL

    输入格式: audio_path\ttranscript
    输出格式: {"source": "audio_path", "target": "transcript"}
    """
    logger.info(f"转换 {txt_file} -> {jsonl_file}")

    count = 0
    with open(txt_file, 'r', encoding='utf-8') as f_in, \
         open(jsonl_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 2:
                logger.warning(f"跳过格式错误的行: {line}")
                continue

            audio_path, transcript = parts

            # 构建 JSONL 格式
            data = {
                "source": audio_path,
                "target": transcript
            }

            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            count += 1

    logger.info(f"转换完成: {count} 条数据")


def main():
    """主函数"""
    project_root = Path("D:/NPU_works/语音/ATC-paraformer")
    data_dir = project_root / "finetune" / "data"

    # 转换训练集
    train_txt = data_dir / "train.txt"
    train_jsonl = data_dir / "train.jsonl"
    txt_to_jsonl(train_txt, train_jsonl)

    # 转换验证集
    val_txt = data_dir / "val.txt"
    val_jsonl = data_dir / "val.jsonl"
    txt_to_jsonl(val_txt, val_jsonl)

    # 转换测试集
    test_txt = data_dir / "test.txt"
    test_jsonl = data_dir / "test.jsonl"
    txt_to_jsonl(test_txt, test_jsonl)

    logger.info("所有数据转换完成！")


if __name__ == "__main__":
    main()
