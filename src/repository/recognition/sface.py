from ._recognition_base import RecognizerBase, get_selected_model_path
import cv2

class Recognizer(RecognizerBase):
    def __init__(self):
        from src.config import ConfigManager
        model_path = get_selected_model_path()
        config = ConfigManager.get_config()
        self.recognizer = cv2.FaceRecognizerSF.create(model_path, "")

            
    
    def alignCrop(self, src_img: cv2.typing.MatLike,
                  face_box: cv2.typing.MatLike)-> cv2.typing.MatLike:
        return self.recognizer.alignCrop(src_img, face_box)
        
        
    def feature(self, aligned_img: cv2.typing.MatLike)->cv2.typing.MatLike:
        return self.recognizer.feature(aligned_img)
    
    
    def match(self, face_feature1:cv2.typing.MatLike ,
            face_feature2: cv2.typing.MatLike,
            dis_type: int = cv2.FaceRecognizerSF_FR_COSINE):
        return self.recognizer.match(face_feature1, face_feature2, dis_type)
