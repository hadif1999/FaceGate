from typing import Literal
import datetime as dt
from pydantic import BaseModel, Field
from uuid import UUID

class QueueMsgSchema(BaseModel):
    uuid: UUID
    msg_type: Literal["RECOGNITION", "REGISTRATION", "CHECK_CAMERA", "CAMERA_STATUS"]
    direction: Literal["incoming", "outgoing", "broadcast"]
    cam_id: int
    face_id: int | None = None
    status: bool | None = None
    message: str | None = None
    camera_uri: str | None = None
    create_date: dt.datetime = Field(default_factory=dt.datetime.now)
    
    
