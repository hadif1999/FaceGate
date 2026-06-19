# Build on Windows:
#   uv run --with pyinstaller pyinstaller --clean --noconfirm test_ws_server_gui.spec
#
# Build on Linux:
#   uv run --with pyinstaller pyinstaller --clean --noconfirm test_ws_server_gui.spec
#
# This creates a one-file GUI executable for scripts/test_ws_server_gui.py.

from PyInstaller.utils.hooks import collect_submodules


hiddenimports = []
hiddenimports += collect_submodules("websockets")


a = Analysis(
    ["scripts/test_ws_server_gui.py"],
    pathex=[],
    binaries=[],
    datas=[],
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
    name="test_ws_server_gui",
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
