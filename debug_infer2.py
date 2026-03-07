"""调试推理输出"""
import json, sys, io, torch, yaml, shutil
from pathlib import Path
from funasr import AutoModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

output_dir = Path('D:/NPU_works/ATC/ATC-paraformer/finetune/output')
project_root = Path('D:/NPU_works/ATC/ATC-paraformer')

cfg_path = output_dir / 'config.yaml'
bak_path = output_dir / '_cfg2.bak'
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
cfg.pop('init_param', None)
cfg.pop('train_data_set_list', None)
cfg.pop('valid_data_set_list', None)
shutil.copy2(cfg_path, bak_path)
with open(cfg_path, 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f, allow_unicode=True)

try:
    model = AutoModel(model=str(output_dir), device='cuda:0', disable_update=True)
finally:
    shutil.copy2(bak_path, cfg_path)
    bak_path.unlink()

ckpt = torch.load(str(output_dir / 'model.pt.avg3'), map_location='cuda:0')
sd = ckpt['state_dict']
res = model.model.load_state_dict(sd, strict=False)
print(f'load_state_dict: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}')

with open(str(project_root / 'data_splits/test.json'), 'r', encoding='utf-8') as f:
    samples = json.load(f)

old_root = r'D:\NPU_works\语音\ATC-paraformer'
new_root = str(project_root)

for s in samples[:5]:
    audio = s['audio'].replace(old_root, new_root)
    if not Path(audio).exists():
        print(f'NOT FOUND: {audio}')
        continue
    gt = s['gt']
    result = model.generate(input=audio, batch_size=1)
    print(f'GT:  {gt}')
    print(f'RES: {result}')
    if isinstance(result, list) and result:
        item = result[0]
        if isinstance(item, dict):
            hyp = item.get('text', '')
            print(f'HYP: {hyp}')
            print(f'HYP repr: {repr(hyp)}')
    print()
