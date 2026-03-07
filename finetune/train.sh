#!/bin/bash
# ATC Paraformer 微调脚本（对齐 seaco_paraformer_large 架构）
#
# 修复说明：
#   - 使用 train_asr_sanm_paraformer.yaml（SANMEncoder+CifPredictorV3，与 seaco 架构一致）
#   - 路径动态推导，无硬编码绝对路径，可直接在服务器运行
#   - FunASR 通过 python -m funasr.bin.train 调用（无需克隆 FunASR 源码）
#   - 输出写入 output_sanm/ 避免覆盖已有的 output/

# GPU 配置
export CUDA_VISIBLE_DEVICES="0"
gpu_num=1

# 路径（相对于本脚本所在目录动态推导，服务器无需修改）
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname "${script_dir}")"
finetune_dir="${script_dir}"

# 数据路径
data_dir="${finetune_dir}/data"
train_data="${data_dir}/train.jsonl"
val_data="${data_dir}/val.jsonl"

# 预训练模型路径
model_dir="${finetune_dir}/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
init_param="${model_dir}/model.pt"
tokens="${model_dir}/tokens.json"
cmvn_file="${model_dir}/am.mvn"

# 输出路径（新目录，避免覆盖原来发散的 output/）
output_dir="${finetune_dir}/output_sanm"
log_file="${output_dir}/train.log"

# 配置文件（对齐 seaco 架构的新 config）
config_name="train_asr_sanm_paraformer.yaml"

mkdir -p "${output_dir}"
echo "开始训练..."
echo "project_root: ${project_root}"
echo "output_dir:   ${output_dir}"
echo "日志文件:     ${log_file}"

# 训练命令
# 注：init_param 指向 seaco model.pt，FunASR 会按 key 名匹配加载
#     SANMEncoder(sanm) 的 key 与 seaco 完全一致，可正确初始化
torchrun \
  --nnodes 1 \
  --nproc_per_node ${gpu_num} \
  -m funasr.bin.train \
  --config-path "${finetune_dir}/conf" \
  --config-name "${config_name}" \
  ++train_data_set_list="${train_data}" \
  ++valid_data_set_list="${val_data}" \
  ++tokenizer_conf.token_list="${tokens}" \
  ++frontend_conf.cmvn_file="${cmvn_file}" \
  ++init_param="${init_param}" \
  ++output_dir="${output_dir}" \
  ++device="cuda" \
  &> "${log_file}"

echo "训练完成！"
echo "查看日志: cat ${log_file}"
