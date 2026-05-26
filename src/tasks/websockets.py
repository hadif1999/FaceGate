import asyncio
import json
import pathlib
import queue
from typing import Any, Literal
from uuid import uuid4

import multiprocessing as mp
import websockets
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from src.config import ConfigManager
from src.repository.database.controller import FaceDatabase
from src.tasks.queue import QueueMsgSchema
from src.utils.utils import find_cam_idx_by_ip, restart_app


RecognizerRuntime = dict[int, tuple[mp.Process, mp.Queue, mp.Queue]]


class WsMsgSchema(BaseModel):
    Type: Literal[
        "reg",
        "del",
        "getList",
        "countDB",
        "delAll",
        "connection",
        "checkCam",
        "getDB",
        "restoreDB",
        "face",
    ]
    memberID: int | None = None
    camIP: str | None = None
    address: str | None = Field(default=None, alias="Address")

    model_config = {"populate_by_name": True}


def _response(**payload) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _json_response(**payload) -> str:
    return json.dumps(_response(**payload), default=str)


def _camera_uri(cam_id: int) -> str | None:
    config = ConfigManager.get_config()
    if 0 <= cam_id < len(config.cameras):
        return config.cameras[cam_id].uri
    return None


def _get_runtime_entry(recognizers: RecognizerRuntime, cam_id: int):
    return recognizers.get(cam_id, (None, None, None))


def queue_event_to_ws_payload(event: QueueMsgSchema) -> dict[str, Any] | None:
    match event.msg_type:
        case "REGISTRATION":
            return _response(
                Type="reg",
                memberID=event.member_id,
                camID=event.cam_id,
                status=event.status,
                ErrorNum=event.error_num,
                similarMemberID=event.similar_member_id,
                confidence=event.confidence,
                message=event.message,
            )
        case "RECOGNITION":
            return _response(
                Type="face",
                memberID=event.member_id,
                camID=event.cam_id,
                confidence=event.confidence,
                status=True,
            )
        case "CAMERA_STATUS":
            return _response(
                Type="checkCam",
                IP=event.camera_uri or _camera_uri(event.cam_id),
                camID=event.cam_id,
                status=event.status,
                message=event.message,
            )
        case "ERROR":
            return _response(
                Type="error",
                camID=event.cam_id,
                memberID=event.member_id,
                status=False,
                message=event.message,
            )
    return None


async def process_recognizer_queues(
    recognizers: RecognizerRuntime,
    outbound_ws_queue: asyncio.Queue[dict[str, Any]],
    interval_sec: float = 0.05,
):
    while True:
        for _, (_, _, out_queue) in list(recognizers.items()):
            while True:
                try:
                    raw_event = out_queue.get_nowait()
                except queue.Empty:
                    break

                try:
                    event = QueueMsgSchema.model_validate(raw_event)
                except Exception as e:
                    logger.error(f"invalid recognizer queue payload: {e}")
                    continue

                if event.direction != "outgoing":
                    logger.warning(f"ignoring non-outgoing recognizer event: {event}")
                    continue

                payload = queue_event_to_ws_payload(event)
                if payload is not None:
                    await outbound_ws_queue.put(payload)
        await asyncio.sleep(interval_sec)


def handle_msg(msg: dict | str, recognizers: RecognizerRuntime) -> None | str:
    if isinstance(msg, str):
        try:
            msg = json.loads(msg)
        except json.JSONDecodeError:
            return _json_response(Type="unknown", status=False, message="invalid json")

    try:
        msg_val = WsMsgSchema.model_validate(msg)
    except ValidationError as e:
        logger.error(f"validation error at input cmd: {e}")
        return _json_response(Type=msg.get("Type", "unknown") if isinstance(msg, dict) else "unknown", status=False, message="invalid command payload")

    config = ConfigManager.get_config()
    db = FaceDatabase(config.vision_setting.face_DB_path)

    try:
        match msg_val.Type:
            case "reg":
                if msg_val.memberID is None:
                    return _json_response(Type="reg", status=False, message="memberID is required")
                cam_idx = find_cam_idx_by_ip(msg_val.camIP, config.cameras)
                if cam_idx is None:
                    return _json_response(Type="reg", memberID=msg_val.memberID, status=False, message="camera not found")
                _, in_queue, _ = _get_runtime_entry(recognizers, cam_idx)
                if in_queue is None:
                    return _json_response(Type="reg", memberID=msg_val.memberID, status=False, message="camera queue not found")

                face_id = db.create_pending_face(msg_val.memberID)
                in_queue.put(
                    QueueMsgSchema(
                        uuid=uuid4(),
                        msg_type="REGISTERING",
                        direction="incoming",
                        cam_id=cam_idx,
                        face_id=face_id,
                        member_id=msg_val.memberID,
                    ).model_dump(),
                    timeout=1,
                )
                return None

            case "checkCam":
                if msg_val.camIP:
                    cam_idx = find_cam_idx_by_ip(msg_val.camIP, config.cameras, use_role_if_not_found=False)
                    if cam_idx is None:
                        return _json_response(Type="checkCam", IP=msg_val.camIP, status=False, message="camera not found")
                    _, in_queue, _ = _get_runtime_entry(recognizers, cam_idx)
                    if in_queue is None:
                        return _json_response(Type="checkCam", IP=msg_val.camIP, status=False, message="camera queue not found")
                    in_queue.put(
                        QueueMsgSchema(uuid=uuid4(), msg_type="CHECK_CAMERA", direction="incoming", cam_id=cam_idx).model_dump(),
                        timeout=1,
                    )
                    return None

                for cam_id, (_, in_queue, _) in recognizers.items():
                    in_queue.put(
                        QueueMsgSchema(uuid=uuid4(), msg_type="CHECK_CAMERA", direction="incoming", cam_id=cam_id).model_dump(),
                        timeout=1,
                    )
                return None

            case "connection":
                return _json_response(Type="connection", status=True)

            case "del":
                if msg_val.memberID is None:
                    return _json_response(Type="del", status=False, message="memberID is required")
                deleted = db.delete_by_member_id(msg_val.memberID)
                return _json_response(Type="del", memberID=msg_val.memberID, status=deleted)

            case "getList":
                return _json_response(Type="getList", status=True, members=db.list_members())

            case "countDB":
                return _json_response(Type="countDB", status=True, num=db.count_members())

            case "delAll":
                num = db.count_members()
                db.del_all()
                return _json_response(Type="delAll", status=True, num=num)

            case "getDB":
                if not msg_val.address:
                    return _json_response(Type="getDB", status=False, message="Address is required")
                backup_path = db.backup_database(msg_val.address, ConfigManager.get_config_path(False))
                return _json_response(Type="getDB", status=True, address=str(backup_path))

            case "restoreDB":
                if not msg_val.address:
                    return _json_response(Type="restoreDB", status=False, message="Address is required")
                db.restore_database(pathlib.Path(msg_val.address))
                return _json_response(Type="restoreDB", status=True, message="database restored; restarting service")

            case "face":
                return None
    except FileNotFoundError as e:
        return _json_response(Type=msg_val.Type, status=False, message=str(e))
    except queue.Full:
        return _json_response(Type=msg_val.Type, status=False, message="camera command queue is full")
    except Exception as e:
        logger.exception(f"failed handling websocket command {msg_val.Type}: {e}")
        return _json_response(Type=msg_val.Type, status=False, message=str(e))


async def _sender(ws, outbound_ws_queue: asyncio.Queue[dict[str, Any]]):
    while True:
        payload = await outbound_ws_queue.get()
        await ws.send(json.dumps(payload, default=str))


async def _receiver(ws, recognizers: RecognizerRuntime, outbound_ws_queue: asyncio.Queue[dict[str, Any]]):
    async for raw_msg in ws:
        response = handle_msg(raw_msg, recognizers)
        if response is not None:
            try:
                payload = json.loads(response)
            except json.JSONDecodeError:
                payload = {"Type": "error", "status": False, "message": response}
            await outbound_ws_queue.put(payload)
            if payload.get("Type") == "restoreDB" and payload.get("status") is True:
                restart_app()


async def main_ws_loop(recognizers: RecognizerRuntime):
    config = ConfigManager.get_config()
    outbound_ws_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
    queue_task = asyncio.create_task(process_recognizer_queues(recognizers, outbound_ws_queue))

    try:
        while True:
            try:
                async with websockets.connect(config.websocket_server.url) as ws:
                    logger.success(f"connected websocket client to {config.websocket_server.url}")
                    sender_task = asyncio.create_task(_sender(ws, outbound_ws_queue))
                    receiver_task = asyncio.create_task(_receiver(ws, recognizers, outbound_ws_queue))
                    done, pending = await asyncio.wait(
                        {sender_task, receiver_task},
                        return_when=asyncio.FIRST_EXCEPTION,
                    )
                    for task in pending:
                        task.cancel()
                    for task in done:
                        task.result()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"websocket disconnected: {e}")
                await asyncio.sleep(config.websocket_server.reconnect_interval_sec)
    finally:
        queue_task.cancel()
        await asyncio.gather(queue_task, return_exceptions=True)
