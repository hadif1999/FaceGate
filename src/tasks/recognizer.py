from src.repository.database.controller import FaceDatabase
from src.repository.recognition._recognition_base import RecognizerBase
from src.repository.detection._detection_base import DetectorBase
from src.config import ConfigManager
import multiprocessing as mp
import cv2
from loguru import logger
from src.errors import InvalidEnviromentError
import numpy as np
from typing import Literal, Tuple
import time
from src.utils.utils import log_exec_time
from src.utils.utils import read_frame, shift_point_percent, crop_percent, crop_by_corners
from .queue import QueueMsgSchema
import datetime as dt
import queue
import random
from src.config import AppConfig
from uuid import uuid4


def _get_detector_cls(input_size: tuple[int, int]) -> DetectorBase:
    from src.config import ConfigManager
    config = ConfigManager.get_config()
    model_name = config.vision_setting.detection.model_name
    match model_name:
        case "yunet":
            try:
                from src.repository.detection.yunet import Detector
                logger.debug(f"loading yunet detector with input_size={input_size}")
                return Detector(input_size)
            except Exception as e:
                logger.critical(f"failed to instantiate yunet detector: {e}")
                raise
        case _:
            raise ValueError(f"unknown detector model type: {model_name!r}")


def _get_recognizer_cls() -> RecognizerBase:
    from src.config import ConfigManager
    config = ConfigManager.get_config()
    model_name = config.vision_setting.recognition.model_name
    match model_name:
        case "sface":
            try:
                from src.repository.recognition.sface import Recognizer
                logger.debug("loading sface recognizer")
                return Recognizer()
            except Exception as e:
                logger.critical(f"failed to instantiate sface recognizer: {e}")
                raise
        case _:
            raise ValueError(f"unknown recognizer model type: {model_name!r}")


def draw_rect_on_frame(frame: cv2.typing.MatLike, dims: np.ndarray, bgr = (0, 255, 0)):
    try:
        x, y, bw, bh = dims[:4].astype(int)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), bgr, 2)
    except Exception as e:
        logger.warning(f"failed to draw rectangle on frame: {e}")
        


def _init_capture(uri: str):
        # ── camera open ───────────────────────────────────────────────────────────
    logger.info(f"opening camera uri={uri!r}")
    cap = cv2.VideoCapture(uri)
    time.sleep(1)

    if not cap.isOpened():
        cap.release()
        logger.critical(f"could not open camera (uri={uri!r})")
        raise InvalidEnviromentError(f"could not open camera {uri}")

    
    # Force MJPEG encoding (allows higher fps/resolution on many webcams)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Verify actual resolution
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info(f"Actual resolution: {width} x {height}")

    # ── initial frame for input size ──────────────────────────────────────────
    ret, frame = cap.read()
    if not ret or frame is None:
        logger.error("failed to grab initial frame from camera, aborting.")
        cap.release()
        cv2.destroyAllWindows()
        return None
    return cap
    


CAMERA_WIN_NAME = "recognition_window"
prev_face_features: None | np.ndarray = None


def recognizer_loop(camera_uri: str, 
                    mp_queue: mp.Queue,
                    lock,
                    interval: float = 0.001,
                    open_camera_window: bool = False,
                    cam_id: int = 0):
    global CAMERA_WIN_NAME, prev_face_features

    # ── config ────────────────────────────────────────────────────────────────
    config = ConfigManager.get_config()
    
    CROP_DIM = ( config.vision_setting.crop.tl_reduce_pct, 
                config.vision_setting.crop.br_reduce_pct )
    
    cap = _init_capture(camera_uri)
    if cap is None: return

    # ── model + db init ───────────────────────────────────────────────────────
    try:
        db = FaceDatabase(config.vision_setting.face_DB_path)
        logger.info(f"face database loaded from {config.vision_setting.face_DB_path!r}")
    except Exception as e:
        logger.critical(f"failed to initialize FaceDatabase: {e}")
        cap.release()
        raise
    
    ret, frame = cap.read()
    frame = crop_percent(frame, CROP_DIM[0],  CROP_DIM[1])
    input_size = tuple(frame.shape[:2])
    logger.info(f"current resolution: {input_size}")
    try:
        detector: DetectorBase = _get_detector_cls(input_size)
        recognizer: RecognizerBase = _get_recognizer_cls()
    except (ValueError, Exception) as e:
        logger.critical(f"failed to initialize models: {e}")
        cap.release()
        raise
    
    logger.info("recognizer loop started")
    same_face = False
    recognition_conf = config.vision_setting.recognition

    # ---------- safe defaults for this iteration ----------
    face_features = None
    face_id = None
    conf = 0.0
    
    # ── main loop ─────────────────────────────────────────────────────────────
    while True:
        frame = read_frame(cap, skip_n_frames=config.vision_setting.skip_n_frames)
        if frame is None: continue
        
        frame = crop_percent(frame, CROP_DIM[0],  CROP_DIM[1])
        h, w = frame.shape[:2]
        
        try:
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
        except Exception as e:
            logger.error(f"detector failed on frame: {e}")
            time.sleep(interval)
            continue

        # ----- handle no‑face case -----
        if faces is None or len(faces) == 0:
            logger.debug("no faces detected, waiting...")
            same_face = False           # reset same‑face flag when no face present
            # still draw window if needed (just show frame without overlays)
            if open_camera_window:
                cv2.imshow(CAMERA_WIN_NAME+f"_{cam_id}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    logger.info("quit key pressed, stopping loop")
                    break
            # skip all face processing for this frame
            time.sleep(interval)
            continue

        # ----- at least one face found -----
        if len(faces) > 1:
            logger.warning(f"detected {len(faces)} faces, selecting face with biggest area")

        # select face with largest area (w*h)
        face = max(faces, key=lambda f: f[2] * f[3])

        # draw all faces if requested
        if open_camera_window:
            for f in faces:
                draw_rect_on_frame(frame, f)
            draw_rect_on_frame(frame, face, (0, 0, 255))   # highlight the chosen face

        # ── feature extraction ────────────────────────────────────────────────
        try:
            face_align = recognizer.alignCrop(frame, face)
            face_features = recognizer.feature(face_align)
        except Exception as e:
            logger.error(f"feature extraction failed: {e}")
            time.sleep(interval)
            continue
        # ----- same‑face check -----
        if prev_face_features is not None:
            try:
                prev_match = recognizer.match(face_features, prev_face_features)
                same_face = prev_match > 0.95
            except Exception as e:
                logger.warning(f"same-face comparison failed, treating as new face: {e}")
                same_face = False

        # ── recognition (only if not same face) ──────────────────────────────
        if not same_face:
            try:
                face_id, conf = db.recognize_face(
                    face_features,
                    recognition_conf.conf_thresh,
                    recognition_conf.similarity_func
                )
            except Exception as e:
                logger.error(f"db.recognize_face failed: {e}")
                time.sleep(interval)
                continue

            if face_id is None:
                logger.warning("face not recognized — unknown person")
            else:
                logger.info(f"*** new face recognized: id={face_id}, confidence={conf:.4f}")
                data = QueueMsgSchema(uuid=uuid4(), msg_type="RECOGNITION",
                                      direction="outgoing", cam_id=cam_id,
                                      face_id=face_id,
                                      create_date=dt.datetime.now())
                mp_queue.put(data.model_dump(), timeout=5)
                time.sleep(config.vision_setting.recognition.after_recognition_delay)
    
        try:
            data_input: dict = mp_queue.get_nowait()
        except queue.Empty:
            data_input = None
        if data_input:
            data = QueueMsgSchema(**data_input)
            if data.direction == "incoming" and data.msg_type == "REGISTRATION" and data.cam_id == cam_id:
                db.update_face(data.face_id, face_features)
                # send update data status 
                payload = QueueMsgSchema(uuid=uuid4(), msg_type="REGISTRATION", 
                            direction="outgoing", cam_id=cam_id,
                            face_id=data.face_id, create_date=dt.datetime.now() )
                mp_queue.put(payload.model_dump(), timeout=5)
            else:
                logger.error(f"error while checking appropriate queue selected for data={data}")
                

        # ── overlay label on frame (only if window is open) ──────────────────
        if open_camera_window:
            try:
                x, y, bw, bh = face[:4].astype(int)
                label = (
                    f"id: {face_id}  conf: {conf:.2f}"
                    if face_id is not None
                    else "Unknown"
                )
                cv2.putText(
                    frame, label,
                    (x, y + bh + 20),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7, (255, 255, 255), 1,
                )
                cv2.imshow(CAMERA_WIN_NAME+f"_{cam_id}", frame)
            except Exception as e:
                logger.warning(f"failed to render overlay: {e}")

        # ── key bindings (only valid when a face is present) ─────────────────
        if open_camera_window:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):       # register
                # Ensure we have valid face features from current frame
                if face_features is not None:
                    try:
                        # use the same threshold as recognition? for registration we want high confidence
                        face_id_check, _ = db.recognize_face(face_features, 0.95, "cosine")
                        if face_id_check is None:
                            logger.info("registering new face...")
                            new_face_id = db.add_face(face_features, random.randint(10, 1000))
                            logger.success(f"registered new face with id={new_face_id}")
                        else:
                            logger.warning(f"face already registered with id={face_id_check}, skipping")
                    except Exception as e:
                        logger.error(f"face registration failed: {e}")
                else:
                    logger.warning("register key pressed but no face detected")

            elif key == ord('u'):       # update
                if face_features is not None:
                    try:
                        face_id_check, _ = db.recognize_face(face_features, 0.95, "cosine")
                        if face_id_check is not None:
                            logger.info(f"updating face id={face_id_check}...")
                            db.update_face(face_id_check, face_features)
                            logger.success(f"updated face id={face_id_check}")
                        else:
                            logger.warning("update requested but face not found in db")
                    except Exception as e:
                        logger.error(f"face update failed: {e}")
                else:
                    logger.warning("update key pressed but no face detected")

            elif key == ord('d'):       # delete
                if face_features is not None:
                    try:
                        face_id_check, _ = db.recognize_face(face_features, 0.95, "cosine")
                        if face_id_check is not None:
                            logger.info(f"deleting face id={face_id_check}...")
                            db.delete_face(face_id_check)
                            logger.success(f"deleted face id={face_id_check}")
                        else:
                            logger.warning("delete requested but face not found in db")
                    except Exception as e:
                        logger.error(f"face deletion failed: {e}")
                else:
                    logger.warning("delete key pressed but no face detected")

        # ── store features for next iteration (only if we have them) ─────────
        if face_features is not None:
            prev_face_features = face_features

        time.sleep(interval)

    # ── cleanup ───────────────────────────────────────────────────────────────
    logger.info("releasing camera and destroying windows")
    cap.release()
    cv2.destroyAllWindows()
    
    

def init_recognizers(open_camera_window: bool = False, begin_processes: bool = True)-> dict[int, Tuple[mp.Process, mp.Queue]]:
    """inits recognizer process objects and their queues and returns them as a pair in a value of dict 
    which key is camera id.

    Args:
        open_camera_window (bool, optional): _description_. Defaults to False.
        begin_processes (bool, optional): _description_. Defaults to True.

    Returns:
        dict[int, Tuple[mp.Process, mp.Queue]]: key is camera id (idx in config) value is tuple of (process, queue)
    """    
    config = ConfigManager.get_config()
    interval = config.vision_setting.interval_sec
    manager = mp.Manager()
    lock = manager.Lock()
    recognizer_processes = {}
    for i, camera in enumerate(config.cameras):
        queue = manager.Queue(maxsize=30) # separate queue for each process
        process = mp.Process(target=recognizer_loop, 
                            args=(camera.uri, queue, lock,
                                  interval, open_camera_window, i),
                            daemon=True,
                            name=f"recognizer_{i}")
        if begin_processes: process.start()
        recognizer_processes[i] = (process, queue)
    return recognizer_processes
