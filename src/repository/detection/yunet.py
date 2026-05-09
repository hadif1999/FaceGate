from __future__ import annotations
import cv2
import face_recognition
import numpy as np
import pathlib
from repository.detection._detection_base import DetectorBase, get_selected_model_path


class Detector(DetectorBase):
    def __init__(self, input_size: tuple[int, int]):
        from src.config import ConfigManager
        super().__init__(input_size)
        config = ConfigManager.get_config()
        detection_settings = config.vision_setting.detection
        model_path = get_selected_model_path()
        match selected_model:=config.vision_setting.detection.model_name:
            case "yunet":
                self.detector = cv2.FaceDetectorYN.create(model_path, "",
                                                        input_size,
                                                        detection_settings.conf_thresh,
                                                        detection_settings.nms_thresh,
                                                        detection_settings.top_k)
            case _:
                raise ValueError(f"undefined detection model {selected_model}")
                
        
    def setInputSize(self, input_size: tuple[int, int]):
        super().setInputSize(input_size)
        self.detector.setInputSize(input_size)
        
        
    def detect(self, frame: cv2.typing.MatLike):
        _, faces = self.detector.detect(frame)
        return faces
        
    
        
        
    