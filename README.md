# ATC Recognizer

本项目是一个基于 `FunASR + PySide6` 的本地语音识别桌面工具，当前已经完成：

- Phase 1: 命令行单文件识别
- Phase 2: 最小 GUI
- Phase 3: 后台线程、状态、进度、日志
- Phase 4: GUI 导出与基础配置
- Phase 5: 批量识别与历史记录

## 当前能力

- 单文件识别
- 批量文件夹识别
- 导出 `txt / srt / json`
- CPU / GPU 设备选择
- 基础参数配置
- 本地配置保存
- 历史记录保存

## 目录说明

```text
app.py                    GUI 入口
app_cli.py                CLI 入口
controllers/              控制层
engines/                  推理层
exporters/                导出层
models/                   数据模型
ui/                       GUI 界面
utils/                    配置、设备、历史记录等工具
workers/                  后台线程任务
model_store/              本地模型目录
inference/                旧脚本与默认输出目录
```

## 环境要求

- Windows
- Python 3.10
- `ffmpeg` 已加入 `PATH`
- 本地模型放在 `model_store/`

当前开发环境验证版本：

- `funasr==1.3.1`
- `PySide6==6.10.2`
- `torch==2.10.0`

## 安装依赖

基础依赖：

```powershell
pip install -r requirements.txt
```

如果要打包 exe：

```powershell
pip install -r requirements-packaging.txt
```

## 模型目录约定

默认会自动查找以下目录：

- `model_store/paraformer_v1/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- `model_store/speech_fsmn_vad_zh-cn-16k-common-pytorch`
- `model_store/punc_ct-transformer_cn-en-common-vocab471067-large`

## 运行方式

GUI:

```powershell
python app.py
```

CLI:

```powershell
python app_cli.py --audio D:\path\sample.mp3
```

兼容 BAT:

```powershell
inference\infer_from_local_windows.bat D:\path\sample.mp3
```

## GUI 使用说明

### 单文件

1. 点击 `Select Audio`
2. 选择输出目录
3. 点击 `Recognize Single`
4. 识别完成后可导出选中格式

### 批量

1. 点击 `Select Folder`
2. 选择包含音频的文件夹
3. 选择输出目录
4. 勾选导出格式
5. 点击 `Recognize Folder`

批量模式会对文件夹内所有支持格式的音频依次识别并自动导出。

## 输出说明

单文件或批量输出都会落到：

```text
<输出根目录>/<音频文件名>/
  result.txt
  result.srt
  result.json
```

## 本地状态文件

- `.atc_gui_settings.json`: GUI 配置
- `.atc_history.json`: 历史记录

## 打包 exe

先安装打包依赖：

```powershell
pip install -r requirements-packaging.txt
```

然后执行：

```powershell
build_windows_exe.bat
```

默认输出到：

```text
dist\ATCRecognizer\
```

## 打包注意事项

- 当前打包脚本只打包 Python 代码和 GUI 入口
- `model_store/` 不建议直接打进 exe，通常与 exe 同目录分发
- 目标机器仍需可访问本地模型目录
- 目标机器需要 `ffmpeg.exe` 可用，或手动放到环境变量中

## 已知说明

- PowerShell 终端里中文文件名偶尔显示乱码，通常是控制台编码问题，不影响 GUI 或导出文件本身
- 当前未实现说话人识别、热词和模型在线切换
