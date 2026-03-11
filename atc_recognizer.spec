# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

import funasr
from PyInstaller.utils.hooks import collect_submodules

project_root = Path.cwd()

datas = []
if (project_root / "assets").exists():
    datas.append((str(project_root / "assets"), "assets"))

funasr_dir = Path(funasr.__file__).resolve().parent
datas.append((str(funasr_dir / "version.txt"), "funasr"))

hiddenimports = []
hiddenimports += collect_submodules("funasr.tokenizer")
hiddenimports += collect_submodules("funasr.frontends")
hiddenimports += collect_submodules("funasr.models.paraformer")
hiddenimports += collect_submodules("funasr.models.sanm")
hiddenimports += collect_submodules("funasr.models.specaug")
hiddenimports += collect_submodules("funasr.models.ct_transformer")
hiddenimports += collect_submodules("funasr.models.fsmn_vad_streaming")

a = Analysis(
    ["app.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "IPython",
        "jupyter",
        "notebook",
        "tensorboard",
        "tensorflow",
        "matplotlib",
        "pandas",
        "sklearn",
        "numba",
        "llvmlite",
        "umap",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ATCRecognizer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
