@echo off
setlocal EnableExtensions

chcp 65001 >nul

set "script_dir=%~dp0"
for %%I in ("%script_dir%..") do set "project_root=%%~fI"
set "finetune_dir=%project_root%\finetune"
set "funasr_train_py=%project_root%\FunASR\funasr\bin\train.py"

set "CUDA_VISIBLE_DEVICES=0"
set "gpu_num=1"
set "MASTER_ADDR=127.0.0.1"
set "MASTER_PORT=29501"
set "HYDRA_FULL_ERROR=1"
set "WANDB_DISABLED=true"
set "WANDB_MODE=disabled"
set "PYTHONPATH=%project_root%\FunASR;%PYTHONPATH%"

set "data_dir=%finetune_dir%\data"
set "train_data=%data_dir%\train.jsonl"
set "val_data=%data_dir%\val.jsonl"
set "model_dir=%finetune_dir%\speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
set "init_param=%model_dir%\model.pt"
set "tokens=%model_dir%\tokens.json"
set "cmvn_file=%model_dir%\am.mvn"

set "output_dir=%finetune_dir%\output_sanm"
set "log_file=%output_dir%\train.log"
set "config_name=train_asr_sanm_paraformer.yaml"

if not exist "%output_dir%" mkdir "%output_dir%"

echo Start training...
echo project_root: %project_root%
echo config_name : %config_name%
echo log_file    : %log_file%

if not exist "%funasr_train_py%" (
  echo ERROR: train.py not found: %funasr_train_py%
  exit /b 1
)

python "%funasr_train_py%" ^
  --config-path "%finetune_dir%\conf" ^
  --config-name "%config_name%" ^
  "++train_data_set_list=%train_data%" ^
  "++valid_data_set_list=%val_data%" ^
  "++tokenizer_conf.token_list=%tokens%" ^
  "++frontend_conf.cmvn_file=%cmvn_file%" ^
  "++dataset_conf.num_workers=0" ^
  "++init_param=%init_param%" ^
  "++output_dir=%output_dir%" ^
  "++device=cuda" ^
  > "%log_file%" 2>&1

if errorlevel 1 (
  echo Training failed. Check log:
  echo type "%log_file%"
  exit /b 1
)

echo Training command started successfully.
echo Check log:
echo type "%log_file%"
