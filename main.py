from __future__ import annotations

import time

import cv2

from repository.database.controller import FaceDatabase
from repository.detection.yunet import FaceDetector
from loguru import logger
from src.errors import CameraError, InvalidEnviromentError


ANALYSIS_INTERVAL_SECONDS = 2.0
CAMERA_INDEX = 0
RECOGNITION_TOLERANCE = 0.6


def draw_result(frame, location, label, color):
    top, right, bottom, left = location
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(
        frame,
        label,
        (left, max(20, top - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def main():
    detector = FaceDetector()
    db = FaceDatabase()
    from src.config import ConfigManager
    config = ConfigManager.get_config()

    cap = cv2.VideoCapture(config.cameras[CAMERA_INDEX].uri)

    if not cap.isOpened():
        logger.error("Error: Cannot open camera")
        raise CameraError("error while opening camera")

    logger.success("Face recognition system started")
    logger.info("Analyzing one frame every 2 seconds")
    logger.info("Press 'q' to quit")

    last_analysis_time = 0.0
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Error: Cannot read frame")
            break

        current_time = time.monotonic()

        if current_time - last_analysis_time >= ANALYSIS_INTERVAL_SECONDS:
            last_analysis_time = current_time
            last_result = None

            encoding, location = detector.get_face_encoding(frame)

            # If nobody is in front of the camera, skip recognition and wait.
            if encoding is not None and location is not None:
                name, confidence = db.recognize_face(
                    encoding, tolerance=RECOGNITION_TOLERANCE
                )

                if name is not None and confidence is not None:
                    label = f"{name} ({confidence:.2f})"
                    color = (0, 255, 0)
                    print(f"Recognized: {label}")
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
                    print("Unknown person")

                last_result = (location, label, color)

        if last_result is not None:
            draw_result(frame, *last_result)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
