import cv2
import pathlib
import sys
from src.config import ConfigManager


class RecognizerBase:
    def __init__(self):
        pass 
    
    
    def alignCrop(self, src_img: cv2.typing.MatLike,
                  face_box: cv2.typing.MatLike)-> cv2.typing.MatLike:
        pass
        
        
    def feature(self, aligned_img: cv2.typing.MatLike)->cv2.typing.MatLike:
        pass
    
    
    def match(self, face_feature1:cv2.typing.MatLike ,
            face_feature2: cv2.typing.MatLike,
            dis_type: int = cv2.FaceRecognizerSF_FR_COSINE):
        pass
    
    
def get_selected_model_path():
    config = ConfigManager.get_config()
    configured_base_path = pathlib.Path(config.vision_setting.models_path)
    candidate_base_paths = [configured_base_path]

    if not configured_base_path.is_absolute():
        config_dir = ConfigManager.get_config_dir(False)
        if config_dir is not None:
            candidate_base_paths.append(config_dir / configured_base_path)

        bundled_root = getattr(sys, "_MEIPASS", None)
        if bundled_root is not None:
            candidate_base_paths.append(pathlib.Path(bundled_root) / configured_base_path)

    match selected_model:=config.vision_setting.recognition.model_name:
        case "sface":
            sface_model_name = "face_recognition_sface_2021dec.onnx"
            for base_path in candidate_base_paths:
                model_path = base_path / selected_model / sface_model_name
                if model_path.exists():
                    return model_path
            model_path = candidate_base_paths[-1] / selected_model / sface_model_name
        case _:
            raise ValueError(f"undefined recognition model {selected_model}")
    
    return model_path
