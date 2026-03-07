import json
import os
from pathlib import Path
import librosa
from sklearn.model_selection import train_test_split

# 使用相对路径
base_dir = Path(__file__).parent.parent  # ATC-paraformer 根目录
wav_dir = base_dir / "chinese_ATC_formatted" / "WAVdata"
txt_dir = base_dir / "chinese_ATC_formatted" / "TXTdata"
output_dir = base_dir / "finetune" / "data"

output_dir.mkdir(parents=True, exist_ok=True)

# 收集所有数据
data_list = []

for txt_file in txt_dir.rglob("*.txt"):
    if txt_file.name == "wordlist.txt":
        continue

    # 构建对应的 wav 文件路径
    relative_path = txt_file.relative_to(txt_dir)
    wav_file = wav_dir / relative_path.with_suffix('.wav')

    if wav_file.exists():
        # 读取文本
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        if text:
            # 使用相对于项目根目录的路径
            relative_wav = str(wav_file.relative_to(base_dir)).replace('\\', '/')

            # 获取音频长度
            try:
                duration = librosa.get_duration(path=str(wav_file))
            except:
                duration = 1.0

            data_list.append({
                "source": str(wav_file.resolve()),  # FunASR 需要绝对路径
                "target": text,
                "source_len": duration
            })

print(f"Total samples: {len(data_list)}")

# 划分 8:1:1 (train / val / test)
train_data, tmp = train_test_split(data_list, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(tmp, test_size=0.5, random_state=42)

print(f"Train samples: {len(train_data)}")
print(f"Val   samples: {len(val_data)}")
print(f"Test  samples: {len(test_data)}")

# 写入 JSONL 文件
with open(output_dir / "train.jsonl", 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open(output_dir / "val.jsonl", 'w', encoding='utf-8') as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open(output_dir / "test.jsonl", 'w', encoding='utf-8') as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 同时把测试集存一份 JSON（eval 脚本用）
test_json_path = output_dir.parent.parent / "data_splits" / "test.json"
test_json_path.parent.mkdir(parents=True, exist_ok=True)
with open(test_json_path, 'w', encoding='utf-8') as f:
    json.dump(
        [{"audio": item["source"], "gt": item["target"]} for item in test_data],
        f, ensure_ascii=False, indent=2
    )

print("Done!")
print(f"  train.jsonl : {output_dir / 'train.jsonl'}")
print(f"  val.jsonl   : {output_dir / 'val.jsonl'}")
print(f"  test.jsonl  : {output_dir / 'test.jsonl'}")
print(f"  test.json   : {test_json_path}")
print(f"Example: {train_data[0]}")
