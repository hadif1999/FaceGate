# Windows EXE Build

Build the executable on a Windows machine. PyInstaller does not cross-compile a Windows `.exe` reliably from Linux.

## Build

From the project root in PowerShell:

```powershell
.\scripts\build_windows.ps1 -Clean
```

Before building, confirm the spec file exists in the project root:

```powershell
Test-Path .\gym_vision.spec
```

This must print:

```text
True
```

This runs:

```powershell
uv run --with pyinstaller pyinstaller --clean --noconfirm .\gym_vision.spec
```

Output:

```text
dist\gym_vision.exe
dist\config.yaml
```

## Runtime Config

`config.yaml` is intentionally kept outside the one-file executable.

Keep this layout:

```text
dist\
  gym_vision.exe
  config.yaml
```

Edit `dist\config.yaml` to change cameras, websocket URL, database path, FPS limits, crop settings, and recognition thresholds. You do not need to rebuild the `.exe` after editing the config.

Run:

```powershell
.\dist\gym_vision.exe
```

Or use an explicit config path:

```powershell
.\dist\gym_vision.exe --config-path C:\Latika\config.yaml
```

## Bundled Models

The `models\` directory is bundled into the one-file executable. The default config value:

```yaml
vision_setting:
  models_path: models/
```

will use external `models\` if it exists near the config/current directory, otherwise it falls back to the bundled PyInstaller model files.

## Runtime Files

Relative database paths are resolved relative to the config file location. With the default:

```yaml
vision_setting:
  face_DB_path: data/face_embeddings.sqlite3
```

the executable will use:

```text
dist\data\face_embeddings.sqlite3
```

The `data` directory is generated outside the one-file executable at runtime:

```text
dist\
  gym_vision.exe
  config.yaml
  data\
    face_embeddings.sqlite3
    logs\
      debug_YYYY-MM-DD.log
      error_YYYY-MM-DD.log
```

This keeps logs and the SQLite database editable, backup-friendly, and persistent across executable rebuilds.
