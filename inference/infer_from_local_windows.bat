@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "PROJECT_ROOT=%SCRIPT_DIR%\.."
set "MODEL_DIR=%PROJECT_ROOT%\models\paraformer_v1\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
set "AUDIO_DIR=%PROJECT_ROOT%\audios"
set "OUTPUT_DIR=%SCRIPT_DIR%\infer_outputs"

if "%VAD_MODEL%"=="" set "VAD_MODEL=%PROJECT_ROOT%\models\speech_fsmn_vad_zh-cn-16k-common-pytorch"
if "%PUNC_MODEL%"=="" set "PUNC_MODEL=%PROJECT_ROOT%\models\punc_ct-transformer_cn-en-common-vocab471067-large"
if "%MAX_SINGLE_SEGMENT_TIME%"=="" set "MAX_SINGLE_SEGMENT_TIME=30000"
if "%BATCH_SIZE_S%"=="" set "BATCH_SIZE_S=20"
if "%BATCH_SIZE_THRESHOLD_S%"=="" set "BATCH_SIZE_THRESHOLD_S=10"

python "%SCRIPT_DIR%\infer_from_local_windows_vad.py" ^
  --root-dir "%PROJECT_ROOT%" ^
  --model-dir "%MODEL_DIR%" ^
  --audio-dir "%AUDIO_DIR%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --force-device "%FORCE_DEVICE%" ^
  --vad-model "%VAD_MODEL%" ^
  --punc-model "%PUNC_MODEL%" ^
  --max-single-segment-time %MAX_SINGLE_SEGMENT_TIME% ^
  --batch-size-s %BATCH_SIZE_S% ^
  --batch-size-threshold-s %BATCH_SIZE_THRESHOLD_S%

exit /b %ERRORLEVEL%
