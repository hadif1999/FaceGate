from src.repository.database.controller import FaceDatabase
from src.repository.recognition._recognition_base import RecognizerBase
from src.repository.detection._detection_base import DetectorBase
import cv2
from loguru import logger
import asyncio
from src.errors import InvalidEnviromentError
import numpy as np


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


def draw_rect_on_frame(frame: cv2.typing.MatLike, dims: np.ndarray):
    try:
        x, y, bw, bh = dims[:4].astype(int)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    except Exception as e:
        logger.warning(f"failed to draw rectangle on frame: {e}")


CAMERA_WIN_NAME = "recognition_window"
prev_face_features: None | np.ndarray = None


async def recognizer_loop(camera_idx: int = 0, interval: float = 0.001,
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
    try:
        uri = config.cameras[camera_idx].uri
    except (IndexError, AttributeError) as e:
        logger.critical(f"invalid camera index {camera_idx}: {e}")
        raise InvalidEnviromentError(f"invalid camera index {camera_idx}") from e

    logger.info(f"opening camera {camera_idx} at uri={uri!r}")
    cap = cv2.VideoCapture(uri)
    await asyncio.sleep(1)

    if not cap.isOpened():
        cap.release()
        logger.critical(f"could not open camera {camera_idx} (uri={uri!r})")
        raise InvalidEnviromentError(f"could not open camera {camera_idx}")
    
    # Force OpenCV to not buffer frames (Essential for real-time)
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
    # ── main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error("failed to grab frame, stopping loop.")
            break

        h, w = frame.shape[:2]

        try:
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
        except Exception as e:
            logger.error(f"detector failed on frame: {e}")
            await asyncio.sleep(interval)
            continue

        if faces is None or len(faces) == 0:
            logger.debug("no faces detected, waiting...")
            await asyncio.sleep(interval)
            continue

        if len(faces) > 1:
            logger.warning(f"detected {len(faces)} faces, waiting for scene to clear")
            await asyncio.sleep(interval)
            continue

        face = faces[0]
        if open_camera_window:
            draw_rect_on_frame(frame, face)

        # ── feature extraction ────────────────────────────────────────────────
        try:
            face_align = recognizer.alignCrop(frame, face)
            face_features = recognizer.feature(face_align)
        except Exception as e:
            logger.error(f"feature extraction failed: {e}")
            await asyncio.sleep(interval)
            continue

        if prev_face_features is not None:
            try:
                prev_match = recognizer.match(face_features, prev_face_features)
                same_face = prev_match > 0.95
                logger.debug(f"same-face match score: {prev_match:.4f} → same={same_face}")
            except Exception as e:
                logger.warning(f"same-face comparison failed, treating as new face: {e}")
                same_face = False

        # ── recognition ───────────────────────────────────────────────────────
        if not same_face:
            try:
                face_id, conf = db.recognize_face(face_features, 
                                                  recognition_conf.conf_thresh,
                                                  recognition_conf.similarity_func)
            except Exception as e:
                logger.error(f"db.recognize_face failed: {e}")
                await asyncio.sleep(interval)
                continue

            if face_id is None:
                logger.warning("face not recognized — unknown person")
            else:
                logger.info(f"*** new face recognized: id={face_id}, confidence={conf:.4f}")

        # ── overlay label ─────────────────────────────────────────────────────
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

        # ── key bindings ──────────────────────────────────────────────────────
        # NOTE: cv2.waitKey(1) blocks the event loop for up to 1ms per frame.
        if open_camera_window:
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord('q')):  # ESC or q
                logger.info("quit key pressed, stopping loop")
                break

            elif key == ord('r'):  # register
                try:
                    face_id, conf = db.recognize_face(face_features, 0.95, "cosine")
                    if face_id is None:
                        logger.info("registering new face...")
                        new_face_id = db.add_face(face_features)
                        logger.success(f"registered new face with id={new_face_id}")
                    else:
                        logger.warning(f"face already registered with id={face_id}, skipping")
                except Exception as e:
                    logger.error(f"face registration failed: {e}")

            elif key == ord('u'):  # update
                try:
                    face_id, conf = db.recognize_face(face_features, 0.95, "cosine")
                    if face_id is not None:
                        logger.info(f"updating face id={face_id}...")
                        db.update_face(face_id, face_features)
                        logger.success(f"updated face id={face_id}")
                    else:
                        logger.warning("update requested but face not found in db")
                except Exception as e:
                    logger.error(f"face update failed: {e}")

            elif key == ord('d'):  # delete
                try:
                    face_id, conf = db.recognize_face(face_features, 0.95, "cosine")
                    if face_id is not None:
                        logger.info(f"deleting face id={face_id}...")
                        db.delete_face(face_id)
                        logger.success(f"deleted face id={face_id}")
                    else:
                        logger.warning("delete requested but face not found in db")
                except Exception as e:
                    logger.error(f"face deletion failed: {e}")

        prev_face_features = face_features
        await asyncio.sleep(interval)

    # ── cleanup ───────────────────────────────────────────────────────────────
    logger.info("releasing camera and destroying windows")
    cap.release()
    cv2.destroyAllWindows()
