from src.repository.database.controller import FaceDatabase
from src.repository.recognition._recognition_base import RecognizerBase
from src.repository.detection._detection_base import DetectorBase
import cv2
from loguru import logger
import asyncio
from src.errors import InvalidEnviromentError
import numpy as np
from typing import Literal


def _get_detector_cls(input_size: tuple[int, int]) -> DetectorBase:
    from src.config import ConfigManager
    try:
        config = ConfigManager.get_config()
    except Exception as e:
        logger.critical(f"failed to load config: {e}")
        raise

    match model_name := config.vision_setting.detection.model_name:
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
    try:
        config = ConfigManager.get_config()
    except Exception as e:
        logger.critical(f"failed to load config: {e}")
        raise

    match model_name := config.vision_setting.recognition.model_name:
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
        
        
def _read_frame(cap: cv2.VideoCapture, skip_n_frames: int = 5):
    for _ in range(skip_n_frames):
        if not cap.grab():
            break
    ret, frame = cap.retrieve()
    if not ret or frame is None:
        logger.error("failed to grab frame, stopping loop.")
        return None
    return frame
    


CAMERA_WIN_NAME = "recognition_window"
prev_face_features: None | np.ndarray = None


async def recognizer_loop(camera_uri: str, interval: float = 0.001,
                          roles: list[Literal["recognition", "registration"]] = ["recognition", "registration"],
                          open_camera_window: bool = False):
    global CAMERA_WIN_NAME, prev_face_features

    # ── config ────────────────────────────────────────────────────────────────
    from src.config import ConfigManager
    try:
        config = ConfigManager.get_config()
    except Exception as e:
        logger.critical(f"failed to load config, cannot start recognizer loop: {e}")
        raise

    # ── camera open ───────────────────────────────────────────────────────────
    logger.info(f"opening camera uri={camera_uri!r}")
    cap = cv2.VideoCapture(camera_uri)
    await asyncio.sleep(1)

    if not cap.isOpened():
        cap.release()
        logger.critical(f"could not open camera (uri={camera_uri!r})")
        raise InvalidEnviromentError(f"could not open camera {camera_uri}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ── initial frame for input size ──────────────────────────────────────────
    ret, frame = cap.read()
    if not ret or frame is None:
        logger.error("failed to grab initial frame from camera, aborting.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # ── model + db init ───────────────────────────────────────────────────────
    try:
        db = FaceDatabase(config.vision_setting.face_DB_path)
        logger.info(f"face database loaded from {config.vision_setting.face_DB_path!r}")
    except Exception as e:
        logger.critical(f"failed to initialize FaceDatabase: {e}")
        cap.release()
        raise

    input_size = tuple(frame.shape[:2])
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
        frame = _read_frame(cap, skip_n_frames=5)
        if frame is None: continue
        
        h, w = frame.shape[:2]

        try:
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
        except Exception as e:
            logger.error(f"detector failed on frame: {e}")
            await asyncio.sleep(interval)
            continue

        # ----- handle no‑face case -----
        if faces is None or len(faces) == 0:
            logger.debug("no faces detected, waiting...")
            same_face = False           # reset same‑face flag when no face present
            # still draw window if needed (just show frame without overlays)
            if open_camera_window:
                cv2.imshow(CAMERA_WIN_NAME, frame)
            # skip all face processing for this frame
            await asyncio.sleep(interval)
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
            await asyncio.sleep(interval)
            continue

        # ----- same‑face check -----
        if prev_face_features is not None:
            try:
                prev_match = recognizer.match(face_features, prev_face_features)
                same_face = prev_match > 0.95
                logger.debug(f"same-face match score: {prev_match:.4f} → same={same_face}")
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
                await asyncio.sleep(interval)
                continue

            if face_id is None:
                logger.warning("face not recognized — unknown person")
            else:
                logger.info(f"*** new face recognized: id={face_id}, confidence={conf:.4f}")

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
                cv2.imshow(CAMERA_WIN_NAME, frame)
            except Exception as e:
                logger.warning(f"failed to render overlay: {e}")

        # ── key bindings (only valid when a face is present) ─────────────────
        if open_camera_window:
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                logger.info("quit key pressed, stopping loop")
                break
            elif key == ord('r'):       # register
                # Ensure we have valid face features from current frame
                if face_features is not None:
                    try:
                        # use the same threshold as recognition? for registration we want high confidence
                        face_id_check, _ = db.recognize_face(face_features, 0.95, "cosine")
                        if face_id_check is None:
                            logger.info("registering new face...")
                            new_face_id = db.add_face(face_features)
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

        await asyncio.sleep(interval)

    # ── cleanup ───────────────────────────────────────────────────────────────
    logger.info("releasing camera and destroying windows")
    cap.release()
    cv2.destroyAllWindows()
