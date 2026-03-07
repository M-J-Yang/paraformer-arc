"""对比原始 seaco 模型 vs 微调模型输出"""
import json, sys, io, torch, yaml, shutil
from pathlib import Path
from funasr import AutoModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

project_root = Path('D:/NPU_works/ATC/ATC-paraformer')
pretrain_dir = project_root / 'finetune' / 'speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
output_dir = project_root / 'finetune' / 'output'

with open(str(project_root / 'data_splits/test.json'), 'r', encoding='utf-8') as f:
    samples = json.load(f)

old_root = r'D:\NPU_works\语音\ATC-paraformer'
new_root = str(project_root)

# 找3个存在的样本
test_audios = []
for s in samples:
    audio = s['audio'].replace(old_root, new_root)
    if Path(audio).exists():
        test_audios.append((audio, s['gt']))
    if len(test_audios) >= 3:
        break

print('=' * 60)
print('1. 原始 seaco 模型（不加载微调权重）')
print('=' * 60)
seaco_model = AutoModel(model=str(pretrain_dir), device='cuda:0', disable_update=True)
for audio, gt in test_audios:
    result = seaco_model.generate(input=audio, batch_size=1)
    hyp = result[0].get('text', '') if isinstance(result, list) and result else ''
    print(f'GT:  {gt}')
    print(f'HYP: {hyp}')
    print()

print('=' * 60)
print('2. 微调模型（加载 model.pt.avg3）- 从 output dir 建架构')
print('=' * 60)
cfg_path = output_dir / 'config.yaml'
bak_path = output_dir / '_cfg3.bak'
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
cfg.pop('init_param', None)
cfg.pop('train_data_set_list', None)
cfg.pop('valid_data_set_list', None)
shutil.copy2(cfg_path, bak_path)
with open(cfg_path, 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f, allow_unicode=True)

try:
    ft_model = AutoModel(model=str(output_dir), device='cuda:0', disable_update=True)
finally:
    shutil.copy2(bak_path, cfg_path)
    bak_path.unlink()

ckpt = torch.load(str(output_dir / 'model.pt.avg3'), map_location='cuda:0')
sd = ckpt['state_dict']
ft_model.model.load_state_dict(sd, strict=False)

for audio, gt in test_audios:
    result = ft_model.generate(input=audio, batch_size=1)
    hyp = result[0].get('text', '') if isinstance(result, list) and result else ''
    print(f'GT:  {gt}')
    print(f'HYP: {hyp}')
    print()
