@echo off
REM ATC Paraformer 微调脚本 (Windows)
REM 基于 FunASR 官方示例

REM GPU 配置
set CUDA_VISIBLE_DEVICES=0
set gpu_num=1

REM 项目路径
set project_root=D:/NPU_works/语音/ATC-paraformer
set finetune_dir=%project_root%/finetune

REM 数据路径
set data_dir=%finetune_dir%/data
set train_data=%data_dir%/train.jsonl
set val_data=%data_dir%/val.jsonl

REM 预训练模型路径
set model_dir=%finetune_dir%/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
set init_param=%model_dir%/model.pt
set tokens=%model_dir%/tokens.json
set cmvn_file=%model_dir%/am.mvn

REM 输出路径
set output_dir=%finetune_dir%/output
set log_file=%output_dir%/train.log

REM 配置文件
set config_name=train_asr_paraformer.yaml

if not exist "%output_dir%" mkdir "%output_dir%"
echo 开始训练...
echo 日志文件: %log_file%

REM 训练命令
torchrun ^
--nnodes 1 ^
--nproc_per_node %gpu_num% ^
%project_root%/FunASR/funasr/bin/train.py ^
--config-path "%finetune_dir%/conf" ^
--config-name "%config_name%" ^
"++train_data_set_list=%train_data%" ^
"++valid_data_set_list=%val_data%" ^
"++tokenizer_conf.token_list=%tokens%" ^
"++frontend_conf.cmvn_file=%cmvn_file%" ^
"++dataset_conf.batch_size=4" ^
"++dataset_conf.batch_type=example" ^
"++dataset_conf.num_workers=2" ^
"++train_conf.max_epoch=20" ^
"++optim_conf.lr=5e-5" ^
"++init_param=%init_param%" ^
"++output_dir=%output_dir%" > "%log_file%" 2>&1

echo 训练完成！
echo 查看日志: type "%log_file%"
pause
