from src.config import ConfigManager
import pathlib
import sys


class DetectorBase:
    
    def __init__(self, input_size: tuple[int, int]):
        pass
    
    
    def setInputSize(self, input_size: tuple[int, int]):
        pass
        
        
    def detect(self, frame):
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

    match selected_model:=config.vision_setting.detection.model_name:
        case "yunet":
            yunet_model_name = "face_detection_yunet_2023mar.onnx"
            for base_path in candidate_base_paths:
                model_path = base_path / selected_model / yunet_model_name
                if model_path.exists():
                    return model_path
            model_path = candidate_base_paths[-1] / selected_model / yunet_model_name
        case _:
            raise ValueError(f"undefined detection model {selected_model}")
    
    return model_path
