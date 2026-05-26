# Build on Windows:
#   uv run --with pyinstaller pyinstaller --clean --noconfirm gym_vision.spec
#
# This creates dist/gym_vision.exe as a one-file executable.
# config.yaml is intentionally not bundled; keep it beside the exe so it can be edited after build.

from PyInstaller.utils.hooks import collect_submodules


datas = [
    ("models", "models"),
]

hiddenimports = []
hiddenimports += collect_submodules("websockets")
hiddenimports += collect_submodules("pydantic_settings")


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "ipykernel",
        "matplotlib",
        "notebook",
        "pytest",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="gym_vision",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
