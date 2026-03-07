import json
import librosa

def add_source_len(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            wav_path = data['wav']

            # 获取音频长度
            try:
                duration = librosa.get_duration(path=wav_path)
                data['source_len'] = duration
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")
                data['source_len'] = 1.0  # 默认值

            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    print("Processing train.jsonl...")
    add_source_len('data/train.jsonl', 'data/train_with_len.jsonl')

    print("Processing val.jsonl...")
    add_source_len('data/val.jsonl', 'data/val_with_len.jsonl')

    print("Done!")
