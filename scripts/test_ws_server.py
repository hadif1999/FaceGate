import argparse
import asyncio
import json
import signal
import sys
from dataclasses import dataclass, field
from typing import Any

import websockets


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8888


@dataclass
class ServerState:
    clients: set[Any] = field(default_factory=set)
    last_client: Any | None = None


def _format_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def parse_command(raw: str) -> dict[str, Any] | None:
    parts = raw.strip().split()
    if not parts:
        return None

    if raw.lstrip().startswith("{"):
        return json.loads(raw)

    command = parts[0]
    match command:
        case "connection":
            return {"Type": "connection"}
        case "countDB":
            return {"Type": "countDB"}
        case "getList":
            return {"Type": "getList"}
        case "delAll":
            return {"Type": "delAll"}
        case "checkCam":
            payload: dict[str, Any] = {"Type": "checkCam"}
            if len(parts) > 1:
                payload["camIP"] = parts[1]
            return payload
        case "reg":
            if len(parts) < 2:
                raise ValueError("usage: reg <memberID> [camIP]")
            payload = {"Type": "reg", "memberID": int(parts[1])}
            if len(parts) > 2:
                payload["camIP"] = parts[2]
            return payload
        case "del":
            if len(parts) < 2:
                raise ValueError("usage: del <memberID>")
            return {"Type": "del", "memberID": int(parts[1])}
        case "getDB":
            if len(parts) < 2:
                raise ValueError("usage: getDB <directory>")
            return {"Type": "getDB", "Address": parts[1]}
        case "restoreDB":
            if len(parts) < 2:
                raise ValueError("usage: restoreDB <db-file>")
            return {"Type": "restoreDB", "Address": parts[1]}
        case _:
            raise ValueError(f"unknown command: {command}")


async def receiver(ws: Any, state: ServerState) -> None:
    async for message in ws:
        try:
            payload = json.loads(message)
            printable = json.dumps(payload, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            printable = message
        print(f"\n<<< from client\n{printable}", flush=True)


async def handler(ws: Any, state: ServerState) -> None:
    peer = ws.remote_address
    state.clients.add(ws)
    state.last_client = ws
    print(f"client connected: {peer}", flush=True)
    try:
        await receiver(ws, state)
    finally:
        state.clients.discard(ws)
        if state.last_client is ws:
            state.last_client = next(iter(state.clients), None)
        print(f"client disconnected: {peer}", flush=True)


async def send_payload(state: ServerState, payload: dict[str, Any]) -> None:
    if state.last_client is None:
        print("no websocket client is connected yet", flush=True)
        return

    message = _format_json(payload)
    await state.last_client.send(message)
    print(f">>> to client\n{json.dumps(payload, indent=2, ensure_ascii=False)}", flush=True)


async def stdin_loop(state: ServerState) -> None:
    print(
        "commands: connection | countDB | getList | checkCam [ip] | "
        "reg <memberID> [ip] | del <memberID> | delAll | getDB <dir> | "
        "restoreDB <db-file> | raw JSON | quit",
        flush=True,
    )

    while True:
        raw = await asyncio.to_thread(sys.stdin.readline)
        if raw == "":
            return

        raw = raw.strip()
        if raw in {"q", "quit", "exit"}:
            return

        try:
            payload = parse_command(raw)
            if payload is not None:
                await send_payload(state, payload)
        except Exception as e:
            print(f"invalid command: {e}", flush=True)


async def auto_probe(state: ServerState, cam_ip: str | None) -> None:
    while state.last_client is None:
        await asyncio.sleep(0.1)

    commands = [
        {"Type": "connection"},
        {"Type": "countDB"},
        {"Type": "getList"},
        {"Type": "checkCam", **({"camIP": cam_ip} if cam_ip else {})},
    ]

    for payload in commands:
        await send_payload(state, payload)
        await asyncio.sleep(0.5)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Local websocket server for gym_vision client testing.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--auto", action="store_true", help="send connection/count/list/checkCam after client connects")
    parser.add_argument("--cam-ip", default=None, help="camera IP to use with --auto checkCam")
    args = parser.parse_args()

    state = ServerState()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    async with websockets.serve(lambda ws: handler(ws, state), args.host, args.port):
        print(f"test websocket server listening on ws://{args.host}:{args.port}", flush=True)

        tasks = [asyncio.create_task(stdin_loop(state))]
        if args.auto:
            tasks.append(asyncio.create_task(auto_probe(state, args.cam_ip)))

        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait({*tasks, stop_task}, return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        for task in done:
            task.result()


if __name__ == "__main__":
    asyncio.run(main())
