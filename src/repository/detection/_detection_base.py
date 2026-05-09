from src.config import ConfigManager


class DetectorBase:
    
    def __init__(self, input_size: tuple[int, int]):
        pass
    
    
    def setInputSize(self, input_size: tuple[int, int]):
        pass
        
        
    def detect(self, frame):
        pass
    
    
def get_selected_model_path():
    import pathlib
    config = ConfigManager.get_config()
    base_path = pathlib.Path(config.vision_setting.models_path)
    match selected_model:=config.vision_setting.detection.model_name:
        case "yunet":
            yunet_model_name = "face_detection_yunet_2023mar.onnx"
            model_path = base_path / selected_model / yunet_model_name
        case _:
            raise ValueError(f"undefined detection model {selected_model}")
    
    return model_path