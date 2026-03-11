@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

python -m PyInstaller --version >nul 2>nul
if errorlevel 1 (
  echo PyInstaller is not installed.
  echo Run: pip install -r requirements-packaging.txt
  exit /b 1
)

pushd "%SCRIPT_DIR%"
python -m PyInstaller --noconfirm --clean atc_recognizer.spec
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" exit /b %EXIT_CODE%

echo Build complete: %SCRIPT_DIR%\dist\ATCRecognizer
exit /b 0
