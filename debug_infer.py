"""单样本推理调试"""
import torch, yaml, shutil, json
from pathlib import Path
from funasr import AutoModel

output_dir = Path('D:/NPU_works/ATC/ATC-paraformer/finetune/output')
project_root = Path('D:/NPU_works/ATC/ATC-paraformer')

cfg_path = output_dir / 'config.yaml'
bak_path = output_dir / '_cfg.bak'
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

inner = model.model
print('Model type:', inner.__class__.__name__)
print('Training mode:', inner.training)

ckpt = torch.load(str(output_dir / 'model.pt.avg3'), map_location='cuda:0')
sd = ckpt['state_dict']
res = inner.load_state_dict(sd, strict=False)
print(f'load_state_dict: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}')

# 检查几个关键权重值（用于验证权重确实被替换）
for key in ['encoder.encoders.0.self_attn.pos_bias_u', 'decoder.decoders.0.src_attn.linear_out.weight']:
    if key in sd:
        model_val = inner.state_dict()[key]
        ckpt_val = sd[key]
        match = torch.allclose(model_val.cpu(), ckpt_val.cpu())
        print(f'  {key}: match={match}, mean={model_val.mean().item():.6f}')

with open(str(project_root / 'data_splits/test.json'), 'r', encoding='utf-8') as f:
    samples = json.load(f)

for s in samples[:5]:
    audio_path = s['audio'].replace(
        'D:\\NPU_works\\语音\\ATC-paraformer',
        str(project_root)
    )
    if not Path(audio_path).exists():
        print(f'SKIP (not found): {audio_path}')
        continue
    print(f'\nGT:    {s["gt"]}')
    result = model.generate(input=audio_path, batch_size=1)
    if isinstance(result, list) and result:
        hyp = result[0].get('text', '') if isinstance(result[0], dict) else str(result[0])
    else:
        hyp = str(result)
    print(f'HYP:   {hyp}')
