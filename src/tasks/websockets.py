from loguru import logger
from src.tasks.queue import QueueMsgSchema
from pydantic import BaseModel, ValidationError
from typing import Literal
import websockets
from src.config import ConfigManager, Camera
import multiprocessing as mp
from src.utils.utils import find_cam_idx_by_ip, restart_app
from uuid import uuid4
from src.repository.database.controller import FaceDatabase
import datetime as dt
import json
import queue
import time



class WsMsgSchema(BaseModel):
    Type: Literal["reg", "del", "countDB", 
                  "delAll", "connection", "checkCam",
                  "getDB", "restoreDB", "face"]
    memberID: int | None = None
    status: Literal["SUCCESS", "FAILED"]|None = None
    camIP: str|None = None
    num: int|None = None
    address: str|None = None
    


def _iter_queue_entries(queues):
    if isinstance(queues, dict):
        for cam_id, value in queues.items():
            if isinstance(value, tuple):
                process = value[0]
                cam_queue = value[-1]
            else:
                process = None
                cam_queue = value
            yield cam_id, process, cam_queue
    else:
        for cam_id, value in enumerate(queues):
            if isinstance(value, tuple):
                process = value[0]
                cam_queue = value[-1]
            else:
                process = None
                cam_queue = value
            yield cam_id, process, cam_queue


def _get_queue_entry(queues, cam_id: int):
    for entry_cam_id, process, cam_queue in _iter_queue_entries(queues):
        if entry_cam_id == cam_id:
            return process, cam_queue
    return None, None


def _camera_uri(cam_id: int) -> str | None:
    config = ConfigManager.get_config()
    if 0 <= cam_id < len(config.cameras):
        return config.cameras[cam_id].uri
    return None


def _wait_for_camera_status(cam_id: int, cam_queue: mp.Queue, timeout_sec: float = 1.5) -> dict:
    deadline = time.monotonic() + timeout_sec
    deferred: list[dict] = []
    try:
        while time.monotonic() < deadline:
            try:
                raw_data = cam_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                data = QueueMsgSchema(**raw_data)
            except Exception:
                deferred.append(raw_data)
                continue

            if data.direction == "outgoing" and data.msg_type == "CAMERA_STATUS" and data.cam_id == cam_id:
                return {
                    "Type": "checkCam",
                    "IP": data.camera_uri or _camera_uri(cam_id),
                    "camID": cam_id,
                    "status": bool(data.status),
                    "message": data.message,
                }

            if data.direction == "incoming" and data.msg_type == "CHECK_CAMERA" and data.cam_id == cam_id:
                cam_queue.put(raw_data, timeout=0.1)
                time.sleep(0.05)
                continue

            deferred.append(raw_data)
    finally:
        for item in deferred:
            try:
                cam_queue.put(item, timeout=0.1)
            except Exception as e:
                logger.warning(f"failed to restore deferred queue message for cam_id={cam_id}: {e}")

    return {
        "Type": "checkCam",
        "IP": _camera_uri(cam_id),
        "camID": cam_id,
        "status": False,
        "message": "camera status response timed out",
    }


def _check_camera_via_queue(cam_id: int, process, cam_queue: mp.Queue) -> dict:
    if process is not None and not process.is_alive():
        return {
            "Type": "checkCam",
            "IP": _camera_uri(cam_id),
            "camID": cam_id,
            "status": False,
            "message": "recognizer process is not alive",
        }

    try:
        cam_queue.put(
            QueueMsgSchema(
                uuid=uuid4(),
                msg_type="CHECK_CAMERA",
                direction="incoming",
                cam_id=cam_id,
                create_date=dt.datetime.now(),
            ).model_dump(),
            timeout=1,
        )
    except Exception as e:
        return {
            "Type": "checkCam",
            "IP": _camera_uri(cam_id),
            "camID": cam_id,
            "status": False,
            "message": f"failed to enqueue camera check: {e}",
        }

    return _wait_for_camera_status(cam_id, cam_queue)


def _check_cameras(msg_val: WsMsgSchema, queues) -> str:
    config = ConfigManager.get_config()
    if msg_val.camIP:
        cam_idx = find_cam_idx_by_ip(msg_val.camIP, config.cameras, use_role_if_not_found=False)
        if cam_idx is None:
            return json.dumps({
                "Type": "checkCam",
                "IP": msg_val.camIP,
                "status": False,
                "message": "camera not found",
            })
        process, cam_queue = _get_queue_entry(queues, cam_idx)
        if cam_queue is None:
            return json.dumps({
                "Type": "checkCam",
                "IP": msg_val.camIP,
                "camID": cam_idx,
                "status": False,
                "message": "camera queue not found",
            })
        return json.dumps(_check_camera_via_queue(cam_idx, process, cam_queue), default=str)

    results = []
    for cam_id, process, cam_queue in _iter_queue_entries(queues):
        results.append(_check_camera_via_queue(cam_id, process, cam_queue))
    return json.dumps({
        "Type": "checkCam",
        "status": all(item["status"] for item in results),
        "cameras": results,
    }, default=str)



def handle_msg(msg: dict, queues: list[mp.Queue])-> None|str:
    if isinstance(msg, str):
        try:
            msg = json.loads(msg)
        except json.JSONDecodeError as e:
            logger.error(f"invalid json websocket message: {e}")
            return json.dumps({"Type": "unknown", "status": False, "message": "invalid json"})
    try:
        msg_val = WsMsgSchema(**msg)
    except ValidationError as e:
        logger.error(f"validation error at input cmd : {e}")
        return
    config = ConfigManager.get_config()
    db = FaceDatabase(config.vision_setting.face_DB_path)
    match msg_val.Type:
        case "reg": # request registering new face (init by making the record and get face_id later add encoding in recognition process)
            if msg_val.memberID is None:
                return json.dumps({"Type": "reg", "status": "FAILED", "message": "memberID is required"})
            cam_idx = find_cam_idx_by_ip(msg_val.camIP, config.cameras)
            if cam_idx is None:
                logger.error("No camera found with specified ip or registration role")
                return
            _, queue = _get_queue_entry(queues, cam_idx)
            if queue is None:
                return json.dumps({"Type": "reg", "status": "FAILED", "message": "camera queue not found"})
            face_id = db.add_face(encoding=None, member_id=msg_val.memberID)
            queue.put(QueueMsgSchema(uuid=uuid4(), msg_type="REGISTRATION", 
                                     direction="incoming", cam_id=cam_idx,
                                     face_id=face_id, create_date=dt.datetime.now()) )
            return None
        case "del":
            if not msg_val.memberID:
                return 
            db.delete_face(msg_val.memberID)
            return WsMsgSchema(Type="del", status="SUCCESS", memberID=msg_val.memberID).model_dump_json(exclude_none=True)
        case "countDB":
            num = len(db.list_face_data())
            return WsMsgSchema(Type="countDB", status="SUCCESS", num=num).model_dump_json(exclude_none=True)
        case "delAll": 
            num = len(db.list_face_data())
            db.del_all()
            return WsMsgSchema(Type="delAll", status="SUCCESS", num=num).model_dump_json(exclude_none=True)
        case "connection": # health-check
            return WsMsgSchema(Type="connection", status="SUCCESS").model_dump_json(exclude_none=True)
        case "checkCam":
            return _check_cameras(msg_val, queues)
        case "getDB":
            import pathlib
            import shutil
            if not msg_val.address: return
            dst_path = pathlib.Path(msg_val.address)
            db_path = config.vision_setting.face_DB_path
            shutil.copy(db_path, dst_path)
            return WsMsgSchema(Type="getDB", status="SUCCESS").model_dump_json(exclude_none=True)
        case "restoreDB":
            import pathlib
            import shutil
            if not msg_val.address: return
            src_path = pathlib.Path(msg_val.address)
            target_path = config.vision_setting.face_DB_path
            shutil.copy(src_path, target_path)
            return WsMsgSchema(Type="restoreDB", status="SUCCESS").model_dump_json(exclude_none=True)
        case _:
            logger.error(f"undefined input msg with type: {msg_val.Type}")
            return
        
        
        
async def main_ws_loop():
    from src.config import ConfigManager
    import json
    config = ConfigManager.get_config()
    async with websockets.connect(config.websocket_server.url) as ws:
        resp: str = await ws.recv()
        if resp:
            payload = handle_msg(resp)
            await ws.send(payload, text=True)
            if "restoreDB" in payload: # restore backup and restart everything 
                restart_app()
            
        # res_dict = json.loads(response)
        
#         await websocket.send("Hello server!")
#         print(response)        



# # Define your callback functions
# def on_message(ws: websocket.WebSocket, message: dict):
#     """Called when a message is received."""
#     response = handle_msg(message, ???) # toDO
#     if response:
#         if isinstance(response, str):
#             ws.send_text(response)
#         else: logger.error(f"wrong payload type={type(response)}")
    
    


# def on_error(ws: websocket.WebSocket, error):
#     """Called when an error occurs."""
#     logger.error(f"Error: {error}")


# def on_close(ws: websocket.WebSocket, close_status_code, close_msg):
#     """Called when the connection is closed."""
#     client, server = _get_websocket_addresses(ws)
#     logger.info(f"connection closed with ws://{server[0]}:{server[1]} : code={close_status_code}: msg={close_msg}")


# def on_open(ws: websocket.WebSocket):
#     """Called when the connection is established."""
#     client, server = _get_websocket_addresses(ws)
#     logger.success(f"connection stablished to ws://{server[0]}:{server[1]}")    
    
#     # ws.send(subscribe_message) # toDO


# def run_websocket_client(is_debug: bool = False):
#     # Enable debug logging to see the WebSocket handshake and frame details
#     config = ConfigManager.get_config()
#     websocket.enableTrace(is_debug)

#     ws_url = config.websocket_server.url
#     ws = websocket.WebSocketApp(ws_url,
#                                 on_open=on_open,
#                                 on_message=on_message,
#                                 on_error=on_error,
#                                 on_close=on_close)

#     # The run_forever() method starts the WebSocket connection in a separate thread
#     # and keeps it alive.
#     ws.run_forever(ping_interval=3, ping_timeout=30,
#                    reconnect=1)



# # async def _ws_handler(websocket ):
#     client_addr = websocket.remote_address
#     logger.info(f"client connection with addr {client_addr[0]}:{client_addr[1]} requested")
#     match websocket.path:
#         case "/":
#             logger.success(f"client addr {client_addr[0]}:{client_addr[1]} connected to '/' ")
#         case _:
#             logger.error(f"access denied from client {client_addr[0]}:{client_addr[1]} to path='{websocket.path}' ")
#             return
            
#     async for msg in websocket:
        
        


# async def run_ws_server(port: int = 8888, host: str = "0.0.0.0"):
#     async with websockets.serve(_ws_handler, host, port):
#         logger.success(f"started websocket server on ws://{host}:{port}")
#         await asyncio.Future()
