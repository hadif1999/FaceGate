class CameraError(BaseException):
    def __init__(self, msg: str = "Unknown error from camera", camera_id: int = 0):
        self.msg = msg
        self.cam_id = camera_id
        
    def __str__(self):
        return f"{self.msg} @ {self.cam_id}"
    
    
class InvalidEnviromentError(BaseException):
    def __init__(self, msg: str, camera_id: int = 0):
        self.msg = msg
        self.cam_id = camera_id

        
    def __str__(self):
        return f"{self.msg} @ {self.cam_id}"
        