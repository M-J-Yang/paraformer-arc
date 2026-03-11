@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "PROJECT_ROOT=%SCRIPT_DIR%\.."
set "OUTPUT_DIR=%SCRIPT_DIR%\infer_outputs"

if "%~1"=="" (
  echo Usage: %~nx0 AUDIO_PATH [OUTPUT_DIR]
  echo Example: %~nx0 D:\audio\sample.mp3
  exit /b 1
)

set "AUDIO_PATH=%~f1"
if not "%~2"=="" set "OUTPUT_DIR=%~f2"

if "%ASR_MODEL_DIR%"=="" set "ASR_MODEL_DIR=%PROJECT_ROOT%\model_store\paraformer_v1\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
if "%VAD_MODEL%"=="" set "VAD_MODEL=%PROJECT_ROOT%\model_store\speech_fsmn_vad_zh-cn-16k-common-pytorch"
if "%PUNC_MODEL%"=="" set "PUNC_MODEL=%PROJECT_ROOT%\model_store\punc_ct-transformer_cn-en-common-vocab471067-large"
if "%ATC_HOTWORD_WORDLIST%"=="" set "ATC_HOTWORD_WORDLIST=%PROJECT_ROOT%\paraformer-arc\chinese_ATC_formatted\TXTdata\wordlist.txt"
if "%ATC_HOTWORD_VOCAB_FREQ%"=="" set "ATC_HOTWORD_VOCAB_FREQ=%PROJECT_ROOT%\paraformer-arc\chinese_ATC_formatted\TXTdata\extracted_vocab_freq.json"
if "%ATC_TEXT_RULES%"=="" set "ATC_TEXT_RULES=%PROJECT_ROOT%\config\atc_text_rules.json"
if "%EXPORT_FORMATS%"=="" set "EXPORT_FORMATS=txt,srt,json"
if "%MAX_SINGLE_SEGMENT_TIME%"=="" set "MAX_SINGLE_SEGMENT_TIME=30000"
if "%BATCH_SIZE_S%"=="" set "BATCH_SIZE_S=20"
if "%BATCH_SIZE_THRESHOLD_S%"=="" set "BATCH_SIZE_THRESHOLD_S=10"

python "%PROJECT_ROOT%\app_cli.py" ^
  --audio "%AUDIO_PATH%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --asr-model "%ASR_MODEL_DIR%" ^
  --vad-model "%VAD_MODEL%" ^
  --punc-model "%PUNC_MODEL%" ^
  --hotword-wordlist "%ATC_HOTWORD_WORDLIST%" ^
  --hotword-vocab-freq "%ATC_HOTWORD_VOCAB_FREQ%" ^
  --text-rules "%ATC_TEXT_RULES%" ^
  --device "%FORCE_DEVICE%" ^
  --export-formats "%EXPORT_FORMATS%" ^
  --max-single-segment-time %MAX_SINGLE_SEGMENT_TIME% ^
  --batch-size-s %BATCH_SIZE_S% ^
  --batch-size-threshold-s %BATCH_SIZE_THRESHOLD_S%

exit /b %ERRORLEVEL%
