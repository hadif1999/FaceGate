# FaceGate

FaceGate is a local lightweight face recognition service for facilities that identifies members from camera streams and communicates recognition, enrollment, and camera status events through WebSocket.

It is designed to run continuously on-site, keep member data local, and integrate with external facility software through a simple JSON protocol. Although the current example configuration is gym-oriented, the service can be used for any facility that needs local camera-based member or visitor identification.

## Install

Use the GitHub Releases page for installation packages and latest binaries:

[Download FaceGate Releases](https://github.com/hadif1999/FaceGate/releases)

Current release channels:

- `windows-latest`: Windows release archive
- `linux-latest`: Linux release archive

Each release publishes:

- a single zip archive containing:
  - the FaceGate service executable or binary
  - the GUI WebSocket test server executable or binary
  - `config.yaml`

## What FaceGate Does

FaceGate combines local camera processing, a local SQLite database, and a persistent WebSocket client.

Main responsibilities:

- connect to one or more RTSP or local cameras
- detect and recognize faces using OpenCV YuNet and SFace
- store embeddings locally in SQLite
- support member enrollment through WebSocket commands
- report recognition events and camera health to an external WebSocket server
- keep logs and database files outside the executable for easy backup and support

## Runtime Model

FaceGate runs as a local service with these parts:

- one main process for configuration, websocket communication, and process supervision
- one recognizer worker process per camera
- one inbound queue and one outbound queue per recognizer
- a local SQLite database for member records and embeddings
- rotating Loguru log files under `data/logs/`

Key runtime behavior:

- per-camera reconnect when a camera is disconnected
- per-camera recognition cooldown
- asynchronous `checkCam` status reporting
- WebSocket auto-reconnect with keepalive ping support
- external `data/` directory generation beside the config file or packaged executable

## Installation Notes

### Windows

Download and extract the Windows release archive:

```text
facegate-windows.zip
```

Extracted contents:

```text
gym_vision.exe
test_ws_server_gui.exe
config.yaml
```

When the executable starts, it generates runtime files beside the executable:

```text
data/
  face_embeddings.sqlite3
  logs/
```

The Windows packaging flow is documented in [WINDOWS_BUILD.md](WINDOWS_BUILD.md).

### Linux

Download and extract the Linux release archive:

```text
facegate-linux.zip
```

Extracted contents:

```text
gym_vision
test_ws_server_gui
config.yaml
```

Then make the binary executable if needed:

```bash
chmod +x ./gym_vision
```

Runtime `data/` is generated beside the config file in the same way as Windows.

## Configuration

FaceGate is configured through `config.yaml`.

Main config groups:

- `general`: logging and preview window behavior
- `cameras`: camera URIs and which camera is allowed to handle default registration
- `vision_setting`: database path, models path, crop, detection, recognition, and camera reconnect behavior
- `websocket_server`: external server URL and keepalive settings
- `performance`: adaptive FPS and CPU-related controls

Important defaults in the current config:

- WebSocket server: `ws://127.0.0.1:8888`
- local database path: `data/face_embeddings.sqlite3`
- bundled model path: `models/`
- recognition cooldown: `3` seconds

The application resolves relative paths from the config file location. That means `data/face_embeddings.sqlite3` becomes local to the installed runtime directory rather than being stored inside the executable bundle.

## WebSocket Protocol

FaceGate acts as a WebSocket client. Your facility software or test server should accept a client connection from FaceGate.

Supported incoming command types:

- `connection`
- `reg`
- `del`
- `getList`
- `countDB`
- `delAll`
- `checkCam`
- `getDB`
- `restoreDB`

Main outgoing event types:

- `face`
- `reg`
- `checkCam`
- `error`

Example commands:

```json
{"Type":"connection"}
{"Type":"checkCam","camIP":"192.168.1.64"}
{"Type":"reg","memberID":1008,"camIP":"192.168.1.64"}
{"Type":"countDB"}
```

Example responses and events:

```json
{"Type":"connection","status":true}
{"Type":"checkCam","IP":"192.168.1.64","camID":0,"status":true}
{"Type":"reg","memberID":1008,"status":true}
{"Type":"face","memberID":1008,"camID":0,"confidence":0.91,"status":true}
```

## Testing Without Facility Software

Two local test servers are included.

### GUI Test Server

Use the Tkinter GUI server when you want to interact with FaceGate manually and inspect sent and received messages in separate panes.

Run:

```bash
uv run python scripts/test_ws_server_gui.py
```

Or use the packaged release asset:

- Windows: `test_ws_server_gui.exe`
- Linux: `test_ws_server_gui`

Features:

- starts a local WebSocket server on `ws://127.0.0.1:8888`
- buttons for common commands like `connection`, `checkCam`, `reg`, `del`, `countDB`, and backup or restore commands
- raw JSON send box
- separate displays for messages sent to FaceGate and messages received from FaceGate

### CLI Test Server

Use the terminal version for quick protocol testing:

```bash
uv run python scripts/test_ws_server.py
```

## Running From Source

Install dependencies:

```bash
uv sync
```

Start the service:

```bash
uv run python main.py
```

Common options:

```bash
uv run python main.py --config-path config.yaml
uv run python main.py --interval 0.01
uv run python main.py --open-camera-window
```

## Build and Release Branches

The repository uses separate platform branches:

- `windows`: Windows-oriented source and Windows release workflow
- `linux`: Linux-oriented source with merged shared updates and Linux release workflow

On every push:

- the corresponding GitHub Actions workflow builds into `dist/`
- `dist/` is uploaded as a workflow artifact
- a single platform zip archive is published to the GitHub Releases page

Release tags used by automation:

- `windows-latest`
- `linux-latest`

## Project Files

Files most users care about:

- `config.yaml`: runtime configuration
- `main.py`: service entrypoint
- `scripts/test_ws_server_gui.py`: GUI WebSocket test server
- `scripts/test_ws_server.py`: CLI WebSocket test server
- `WINDOWS_BUILD.md`: Windows packaging notes

## Operational Notes

- FaceGate keeps database and logs local to the installed runtime directory.
- Recognition uses local ONNX models through OpenCV YuNet and SFace.
- Camera workers are isolated so one camera failure does not stop the whole service.
- If the WebSocket server is unavailable, FaceGate keeps retrying until it reconnects.
