# FaceGate

FaceGate is a local lightweight face recognition service for facilities that identifies members from camera streams and communicates recognition, enrollment, camera visibility, and camera status events through WebSocket.

It is designed to run continuously on-site, keep member data local, and integrate with external facility software through a simple JSON protocol. The current configuration is gym-oriented, but the service can be used for any facility that needs local camera-based member or visitor identification.

## Installation

The recommended install path is the GitHub Releases page:

[Download FaceGate Releases](https://github.com/hadif1999/FaceGate/releases)

Current moving release channels:

- `windows-latest`
- `linux-latest`

Each release is published as a single zip archive that contains:

- the FaceGate service executable or binary
- the GUI WebSocket test server executable or binary
- `config.yaml`

## What FaceGate Does

FaceGate combines local camera processing, a local SQLite database, and a persistent WebSocket client.

Main responsibilities:

- connect to one or more RTSP, HTTP, local-device, or indexed cameras
- detect and recognize faces using OpenCV YuNet and SFace
- store embeddings locally in SQLite
- support member enrollment through WebSocket commands
- report recognition events and camera health to an external WebSocket server
- show or hide camera preview windows on demand through `camShow`
- restore both database and config from a backup folder, then restart the app
- keep logs and database files outside the executable bundle for easy backup and support

## Runtime Model

FaceGate runs as a local service with these parts:

- one main process for configuration, WebSocket communication, and process supervision
- one recognizer worker process per camera
- one inbound queue and one outbound queue per recognizer
- a local SQLite database for member records and embeddings
- rotating Loguru log files under `data/logs/`

Key runtime behavior:

- per-camera reconnect when a camera is disconnected
- per-camera recognition cooldown
- asynchronous `checkCam` status reporting
- `camShow` can open or close the preview window for one camera
- WebSocket auto-reconnect with keepalive ping support
- backup restore replaces both DB and config from the target folder, then restarts the app
- runtime `data/` directory generation beside the config file or packaged executable

## Configuration

FaceGate is configured through `config.yaml`.

Main config groups:

- `general`: logging level
- `cameras`: camera URIs, registration eligibility, and optional preview window behavior
- `vision_setting`: database path, models path, crop, detection, recognition, and camera reconnect behavior
- `websocket_server`: external server URL and keepalive settings
- `performance`: adaptive FPS and CPU-related controls

Camera URI examples supported by the current code and OpenCV include:

```yaml
cameras:
  # RTSP stream
  - URI: "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101"

  # Another RTSP example
  - URI: "rtsp://192.168.1.64:554/live/ch00_0"

  # HTTP/MJPEG stream
  - URI: "http://192.168.1.64:8080/video"

  # Local file stream
  - URI: "file:///home/user/videos/sample.mp4"

  # Local device path
  - URI: "/dev/video0"

  # Numeric camera index
  - URI: 0

  # String camera index
  - URI: "0"
```

Per-camera preview windows are controlled on the camera entry itself:

```yaml
cameras:
  - URI: "/dev/video0"
    can_register: false
    open_camera_window: true
```

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
- `camShow`
- `getDB`
- `restoreDB`

Main outgoing event types:

- `face`
- `reg`
- `checkCam`
- `camShow`
- `error`

Command notes:

- `reg` creates a pending member record, then sends a `REGISTERING` message into the selected camera queue.
- `checkCam` sends a camera check request into one camera queue, or into all camera queues if no camera address is provided.
- `camShow` toggles the preview window for the target camera; `status: true` opens it and `status: false` closes it.
- `getDB` backs up both the current DB and the active config into the requested folder.
- `restoreDB` expects a folder that contains both the backup DB and `config.yaml`, restores both into the current runtime locations, and then restarts the app.

Example commands:

```json
{"Type":"connection"}
{"Type":"checkCam","camIP":"192.168.1.64"}
{"Type":"camShow","camIP":"192.168.1.64","status":true}
{"Type":"reg","memberID":1008,"camIP":"192.168.1.64"}
{"Type":"countDB"}
```

Example responses and events:

```json
{"Type":"connection","status":true}
{"Type":"checkCam","IP":"192.168.1.64","camID":0,"status":true}
{"Type":"camShow","IP":"192.168.1.64","camID":0,"status":true}
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
- buttons for common commands like `connection`, `checkCam`, `camShow`, `reg`, `del`, `countDB`, `getDB`, and `restoreDB`
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
uv run main.py
```

Useful runtime notes:

- edit `config.yaml` beside the executable or source tree before starting the service
- if you want camera preview windows, set `open_camera_window: true` on the relevant camera entry
- on headless Linux systems, keep preview windows disabled
- if a camera disconnects, the worker keeps retrying and shows a not-responding overlay until the camera returns

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

## Backup and Restore

Backup and restore use a folder-based layout.

When you request a backup with `getDB`:

- the active SQLite database is copied into the target folder
- the active `config.yaml` is copied into the same folder

When you request a restore with `restoreDB`:

- the app looks for both the database and `config.yaml` in the target folder
- both files replace the current in-use files
- the app restarts so the restored config takes effect

## Troubleshooting

- If the app cannot reach the WebSocket server, it keeps retrying automatically.
- If a camera is unavailable, the worker logs the failure and retries without crashing the whole service.
- If preview windows behave badly on Linux, disable `open_camera_window` for that camera.
- If OpenCV is missing in a packaged Windows build, make sure the release zip was extracted fully and the bundled `models/` directory is present.

## Operational Notes

- FaceGate keeps database and logs local to the installed runtime directory.
- Recognition uses local ONNX models through OpenCV YuNet and SFace.
- Logs are written with Loguru and inherit the configured log level across subprocesses.
