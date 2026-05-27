import datetime as dt
import multiprocessing as mp
import platform
import queue
import random
import time
from typing import Tuple
from uuid import uuid4

import cv2
import numpy as np
from loguru import logger

from src.config import AppConfig, ConfigManager
from src.repository.database.controller import FaceDatabase
from src.repository.detection._detection_base import DetectorBase
from src.repository.recognition._recognition_base import RecognizerBase
from src.tasks.queue import QueueMsgSchema
from src.utils.utils import crop_percent, read_frame


def _get_mp_context() -> mp.context.BaseContext:
    return mp.get_context("spawn" if platform.system() == "Windows" else "fork")


def _get_detector_cls(input_size: tuple[int, int]) -> DetectorBase:
    config = ConfigManager.get_config()
    match config.vision_setting.detection.model_name:
        case "yunet":
            from src.repository.detection.yunet import Detector

            logger.debug(f"loading yunet detector with input_size={input_size}")
            return Detector(input_size)
        case model_name:
            raise ValueError(f"unknown detector model type: {model_name!r}")


def _get_recognizer_cls() -> RecognizerBase:
    config = ConfigManager.get_config()
    match config.vision_setting.recognition.model_name:
        case "sface":
            from src.repository.recognition.sface import Recognizer

            logger.debug("loading sface recognizer")
            return Recognizer()
        case model_name:
            raise ValueError(f"unknown recognizer model type: {model_name!r}")


def draw_rect_on_frame(frame: cv2.typing.MatLike, dims: np.ndarray, bgr=(0, 255, 0)):
    try:
        x, y, bw, bh = dims[:4].astype(int)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), bgr, 2)
    except Exception as e:
        logger.warning(f"failed to draw rectangle on frame: {e}")


def calculate_target_fps(camera_count: int) -> float:
    perf = ConfigManager.get_config().performance
    if camera_count <= 1:
        return perf.max_fps_single_camera
    if camera_count <= 3:
        return perf.max_fps_2_3_cameras
    if camera_count <= 5:
        return perf.max_fps_4_5_cameras
    return perf.max_fps_more_cameras


def _resize_for_recognition(frame: cv2.typing.MatLike, max_width: int) -> cv2.typing.MatLike:
    if max_width <= 0:
        return frame

    h, w = frame.shape[:2]
    if w <= max_width:
        return frame

    scale = max_width / float(w)
    new_size = (max_width, max(1, int(h * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _init_capture(uri: str):
    logger.info(f"opening camera uri={uri!r}")
    cap = cv2.VideoCapture(uri)
    time.sleep(1)

    if not cap.isOpened():
        cap.release()
        logger.warning(f"could not open camera (uri={uri!r})")
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info(f"Actual resolution: {width} x {height}")

    return cap


def _release_capture(cap):
    if cap is None:
        return
    try:
        cap.release()
    except Exception as e:
        logger.debug(f"failed releasing camera capture: {e}")


CAMERA_WIN_NAME = "recognition_window"
prev_face_features: None | np.ndarray = None


def _queue_put_safe(out_queue: mp.Queue, payload: QueueMsgSchema):
    try:
        out_queue.put(payload.model_dump(), timeout=1)
    except Exception as e:
        logger.error(f"failed to publish queue message: {e}")


def _camera_status_payload(cam_id: int, camera_uri: str, status: bool, message: str | None = None) -> QueueMsgSchema:
    return QueueMsgSchema(
        uuid=uuid4(),
        msg_type="CAMERA_STATUS",
        direction="outgoing",
        cam_id=cam_id,
        status=status,
        camera_uri=camera_uri,
        message=message,
        create_date=dt.datetime.now(),
    )


def recognizer_loop(
    camera_uri: str,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    lock,
    interval: float = 0.001,
    open_camera_window: bool = False,
    cam_id: int = 0,
    config_snapshot: dict | None = None,
):
    global CAMERA_WIN_NAME, prev_face_features

    if ConfigManager.get_config(False) is None and config_snapshot is not None:
        ConfigManager.update_config(AppConfig.model_validate(config_snapshot))

    config = ConfigManager.get_config()
    crop_dim = (
        config.vision_setting.crop.tl_reduce_pct,
        config.vision_setting.crop.br_reduce_pct,
    )
    performance = config.performance
    try:
        cv2.setNumThreads(max(1, performance.opencv_threads_per_process))
    except Exception as e:
        logger.debug(f"failed to set OpenCV thread count: {e}")

    try:
        db = FaceDatabase(config.vision_setting.face_DB_path)
        logger.info(f"face database loaded from {config.vision_setting.face_DB_path!r}")
    except Exception as e:
        logger.critical(f"failed to initialize FaceDatabase: {e}")
        raise

    logger.info("recognizer loop started")
    cap = None
    detector: DetectorBase | None = None
    recognizer: RecognizerBase | None = None
    same_face = False
    recognition_conf = config.vision_setting.recognition
    base_target_fps = max(calculate_target_fps(len(config.cameras)), 0.1)
    enrollment_target_fps = max(performance.enrollment_fps, base_target_fps)
    target_fps = base_target_fps
    next_frame_at = 0.0
    camera_reconnect_interval = config.vision_setting.camera_reconnect_interval_sec
    camera_status_interval = config.vision_setting.camera_status_interval_sec
    face_features = None
    member_id = None
    conf = 0.0
    last_recognition_sent_at = 0.0
    last_no_face_log_at = 0.0
    last_unknown_face_log_at = 0.0
    last_detector_error_log_at = 0.0
    last_feature_error_log_at = 0.0
    last_camera_status: bool | None = None
    last_camera_status_sent_at = 0.0
    next_camera_retry_at = 0.0
    registering_face_id: int | None = None
    registering_member_id: int | None = None
    registering_started_at: float | None = None

    def publish_camera_status(status: bool, message: str, force: bool = False):
        nonlocal last_camera_status, last_camera_status_sent_at
        now = time.monotonic()
        should_publish = (
            force
            or last_camera_status is None
            or status != last_camera_status
            or now - last_camera_status_sent_at >= camera_status_interval
        )
        if not should_publish:
            return
        _queue_put_safe(out_queue, _camera_status_payload(cam_id, camera_uri, status, message))
        last_camera_status = status
        last_camera_status_sent_at = now

    while True:
        try:
            while True:
                data = QueueMsgSchema(**in_queue.get_nowait())
                if data.direction != "incoming" or data.cam_id != cam_id:
                    logger.warning(f"ignored queue command for this recognizer: {data}")
                    continue

                if data.msg_type == "CHECK_CAMERA":
                    status = bool(cap is not None and cap.isOpened())
                    _queue_put_safe(
                        out_queue,
                        _camera_status_payload(
                            cam_id,
                            camera_uri,
                            status,
                            "camera is open" if status else "camera is not open",
                        ),
                    )
                elif data.msg_type == "REGISTERING":
                    registering_face_id = data.face_id
                    registering_member_id = data.member_id
                    registering_started_at = time.monotonic()
                    next_frame_at = 0.0
                    logger.info(
                        f"registration started cam_id={cam_id} face_id={registering_face_id} "
                        f"member_id={registering_member_id}"
                    )
                else:
                    logger.warning(f"unsupported incoming queue command: {data}")
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"failed to process recognizer command cam_id={cam_id}: {e}")

        if registering_face_id is not None and registering_started_at is not None:
            if time.monotonic() - registering_started_at > recognition_conf.enrollment_timeout_sec:
                try:
                    db.delete_pending_face(registering_face_id)
                except Exception as e:
                    logger.warning(f"failed to delete timed-out pending face_id={registering_face_id}: {e}")
                _queue_put_safe(
                    out_queue,
                    QueueMsgSchema(
                        uuid=uuid4(),
                        msg_type="REGISTRATION",
                        direction="outgoing",
                        cam_id=cam_id,
                        face_id=registering_face_id,
                        member_id=registering_member_id,
                        status=False,
                        error_num=2,
                        message="registration timed out waiting for face",
                        create_date=dt.datetime.now(),
                    ),
                )
                registering_face_id = None
                registering_member_id = None
                registering_started_at = None

        target_fps = enrollment_target_fps if registering_face_id is not None else base_target_fps

        now = time.monotonic()
        if cap is None or not cap.isOpened():
            if now >= next_camera_retry_at:
                try:
                    cap = _init_capture(camera_uri)
                    if cap is None:
                        publish_camera_status(False, "camera unavailable; retrying")
                    else:
                        publish_camera_status(True, "camera connected", force=True)
                        same_face = False
                        prev_face_features = None
                except Exception as e:
                    _release_capture(cap)
                    cap = None
                    publish_camera_status(False, f"camera open failed: {e}")
                next_camera_retry_at = time.monotonic() + camera_reconnect_interval
            time.sleep(max(interval, 0.2))
            continue

        now = time.monotonic()
        if now < next_frame_at:
            time.sleep(min(next_frame_at - now, 0.2))
            continue
        next_frame_at = now + (1.0 / target_fps)

        frame = read_frame(cap, skip_n_frames=config.vision_setting.skip_n_frames)
        if frame is None:
            publish_camera_status(False, "failed to read frame; reconnecting", force=True)
            _release_capture(cap)
            cap = None
            same_face = False
            prev_face_features = None
            next_camera_retry_at = time.monotonic() + camera_reconnect_interval
            time.sleep(max(interval, 0.2))
            continue

        try:
            frame = _resize_for_recognition(frame, performance.frame_resize_width)
            frame = crop_percent(frame, crop_dim[0], crop_dim[1])
        except Exception as e:
            logger.error(f"crop failed: {e}")
            time.sleep(interval)
            continue
        h, w = frame.shape[:2]

        if detector is None or recognizer is None:
            try:
                detector = _get_detector_cls((w, h))
                recognizer = _get_recognizer_cls()
                publish_camera_status(True, "camera ready", force=True)
            except Exception as e:
                logger.error(f"failed to initialize models: {e}")
                _queue_put_safe(
                    out_queue,
                    QueueMsgSchema(
                        uuid=uuid4(),
                        msg_type="ERROR",
                        direction="outgoing",
                        cam_id=cam_id,
                        status=False,
                        message=f"failed to initialize models: {e}",
                        create_date=dt.datetime.now(),
                    ),
                )
                time.sleep(max(camera_reconnect_interval, interval))
                continue

        try:
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
        except Exception as e:
            now = time.monotonic()
            if now - last_detector_error_log_at > 5:
                logger.error(f"detector failed on frame: {e}")
                last_detector_error_log_at = now
            time.sleep(interval)
            continue

        if faces is None or len(faces) == 0:
            now = time.monotonic()
            if now - last_no_face_log_at > 5:
                logger.debug("no faces detected, waiting...")
                last_no_face_log_at = now
            same_face = False
            if open_camera_window:
                cv2.imshow(CAMERA_WIN_NAME + f"_{cam_id}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    logger.info("quit key pressed, stopping loop")
                    break
            time.sleep(interval)
            continue

        if len(faces) > 1:
            logger.warning(f"detected {len(faces)} faces, selecting face with biggest area")
        face = max(faces, key=lambda f: f[2] * f[3])

        if open_camera_window:
            for f in faces:
                draw_rect_on_frame(frame, f)
            draw_rect_on_frame(frame, face, (0, 0, 255))

        try:
            face_align = recognizer.alignCrop(frame, face)
            face_features = recognizer.feature(face_align)
        except Exception as e:
            now = time.monotonic()
            if now - last_feature_error_log_at > 5:
                logger.error(f"feature extraction failed: {e}")
                last_feature_error_log_at = now
            time.sleep(interval)
            continue

        if registering_face_id is not None:
            try:
                similar_member_id, duplicate_conf = db.find_match(
                    face_features,
                    recognition_conf.duplicate_thresh,
                    recognition_conf.similarity_func,
                )
                if similar_member_id is not None and similar_member_id != registering_member_id:
                    db.delete_pending_face(registering_face_id)
                    payload = QueueMsgSchema(
                        uuid=uuid4(),
                        msg_type="REGISTRATION",
                        direction="outgoing",
                        cam_id=cam_id,
                        face_id=registering_face_id,
                        member_id=registering_member_id,
                        status=False,
                        error_num=3,
                        similar_member_id=similar_member_id,
                        confidence=duplicate_conf,
                        message="duplicate face detected",
                        create_date=dt.datetime.now(),
                    )
                else:
                    updated = db.update_face(registering_face_id, face_features)
                    if not updated:
                        raise RuntimeError(
                            f"pending registration row not found for face_id={registering_face_id}"
                        )
                    payload = QueueMsgSchema(
                        uuid=uuid4(),
                        msg_type="REGISTRATION",
                        direction="outgoing",
                        cam_id=cam_id,
                        face_id=registering_face_id,
                        member_id=registering_member_id,
                        status=True,
                        create_date=dt.datetime.now(),
                    )
                _queue_put_safe(out_queue, payload)
            except Exception as e:
                try:
                    db.delete_pending_face(registering_face_id)
                except Exception:
                    logger.warning(f"failed to delete pending face_id={registering_face_id}")
                _queue_put_safe(
                    out_queue,
                    QueueMsgSchema(
                        uuid=uuid4(),
                        msg_type="ERROR",
                        direction="outgoing",
                        cam_id=cam_id,
                        face_id=registering_face_id,
                        member_id=registering_member_id,
                        status=False,
                        message=str(e),
                        create_date=dt.datetime.now(),
                    ),
                )
            registering_face_id = None
            registering_member_id = None
            registering_started_at = None
            time.sleep(interval)
            continue

        if prev_face_features is not None:
            try:
                same_face = recognizer.match(face_features, prev_face_features) > 0.95
            except Exception as e:
                logger.warning(f"same-face comparison failed, treating as new face: {e}")
                same_face = False

        if not same_face:
            try:
                member_id, conf = db.find_match(
                    face_features,
                    recognition_conf.conf_thresh,
                    recognition_conf.similarity_func,
                )
            except Exception as e:
                logger.error(f"db.find_match failed: {e}")
                time.sleep(interval)
                continue

            if member_id is None:
                now = time.monotonic()
                if now - last_unknown_face_log_at > 5:
                    logger.warning("face not recognized - unknown person")
                    last_unknown_face_log_at = now
            else:
                now = time.monotonic()
                if now - last_recognition_sent_at >= recognition_conf.after_recognition_delay:
                    logger.info(f"*** new face recognized: member_id={member_id}, confidence={conf:.4f}")
                    _queue_put_safe(
                        out_queue,
                        QueueMsgSchema(
                            uuid=uuid4(),
                            msg_type="RECOGNITION",
                            direction="outgoing",
                            cam_id=cam_id,
                            member_id=member_id,
                            confidence=conf,
                            create_date=dt.datetime.now(),
                        ),
                    )
                    last_recognition_sent_at = now

        if open_camera_window:
            try:
                x, y, bw, bh = face[:4].astype(int)
                label = f"id: {member_id}  conf: {conf:.2f}" if member_id is not None else "Unknown"
                cv2.putText(frame, label, (x, y + bh + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
                cv2.imshow(CAMERA_WIN_NAME + f"_{cam_id}", frame)
            except Exception as e:
                logger.warning(f"failed to render overlay: {e}")

        if open_camera_window:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r") and face_features is not None:
                try:
                    member_id_check, _ = db.find_match(face_features, 0.95, "cosine")
                    if member_id_check is None:
                        new_face_id = db.add_face(face_features, random.randint(10, 1000))
                        logger.success(f"registered new face with id={new_face_id}")
                    else:
                        logger.warning(f"face already registered with member_id={member_id_check}, skipping")
                except Exception as e:
                    logger.error(f"face registration failed: {e}")
            elif key == ord("u") and face_features is not None:
                try:
                    member_id_check, _ = db.find_match(face_features, 0.95, "cosine")
                    if member_id_check is not None:
                        row = db.get_face_by_member_id(member_id_check)
                        if row is not None and db.update_face(row["id"], face_features, pending_only=False):
                            logger.success(f"updated member_id={member_id_check}")
                except Exception as e:
                    logger.error(f"face update failed: {e}")
            elif key == ord("d") and face_features is not None:
                try:
                    member_id_check, _ = db.find_match(face_features, 0.95, "cosine")
                    if member_id_check is not None:
                        db.delete_by_member_id(member_id_check)
                        logger.success(f"deleted member_id={member_id_check}")
                except Exception as e:
                    logger.error(f"face deletion failed: {e}")

        if face_features is not None:
            prev_face_features = face_features
        time.sleep(interval)

    logger.info("releasing camera and destroying windows")
    _release_capture(cap)
    cv2.destroyAllWindows()


def init_recognizers(open_camera_window: bool = False, begin_processes: bool = True) -> dict[int, Tuple[mp.Process, mp.Queue, mp.Queue]]:
    config = ConfigManager.get_config()
    interval = config.vision_setting.interval_sec
    config_snapshot = config.model_dump(mode="python")
    ctx = _get_mp_context()
    lock = ctx.Lock()
    recognizer_processes = {}
    for i, camera in enumerate(config.cameras):
        in_queue = ctx.Queue(maxsize=30)
        out_queue = ctx.Queue(maxsize=30)
        process = ctx.Process(
            target=recognizer_loop,
            args=(camera.uri, in_queue, out_queue, lock, interval, open_camera_window, i, config_snapshot),
            daemon=True,
            name=f"recognizer_{i}",
        )
        if begin_processes:
            process.start()
        recognizer_processes[i] = (process, in_queue, out_queue)
    return recognizer_processes


def start_recognizer_process(
    cam_id: int,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    lock,
    open_camera_window: bool = False,
) -> mp.Process:
    config = ConfigManager.get_config()
    config_snapshot = config.model_dump(mode="python")
    ctx = _get_mp_context()
    process = ctx.Process(
        target=recognizer_loop,
        args=(
            config.cameras[cam_id].uri,
            in_queue,
            out_queue,
            lock if lock is not None else ctx.Lock(),
            config.vision_setting.interval_sec,
            open_camera_window,
            cam_id,
            config_snapshot,
        ),
        daemon=True,
        name=f"recognizer_{cam_id}",
    )
    process.start()
    return process
