import cv2
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
    import pathlib
    config = ConfigManager.get_config()
    base_path = pathlib.Path(config.vision_setting.models_path)
    match selected_model:=config.vision_setting.recognition.model_name:
        case "sface":
            sface_model_name = "face_recognition_sface_2021dec.onnx"
            model_path = base_path / selected_model / sface_model_name
        case _:
            raise ValueError(f"undefined recognition model {selected_model}")
    
    return model_path