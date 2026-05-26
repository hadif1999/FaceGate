from typing import Literal
import datetime as dt
from pydantic import BaseModel, Field
from uuid import UUID

class QueueMsgSchema(BaseModel):
    uuid: UUID
    msg_type: Literal["REGISTERING", "REGISTRATION", "RECOGNITION", "CHECK_CAMERA", "CAMERA_STATUS", "ERROR"]
    direction: Literal["incoming", "outgoing", "broadcast"]
    cam_id: int
    face_id: int | None = None
    member_id: int | None = None
    status: bool | None = None
    error_num: int | None = None
    similar_member_id: int | None = None
    confidence: float | None = None
    message: str | None = None
    camera_uri: str | None = None
    create_date: dt.datetime = Field(default_factory=dt.datetime.now)
    
    
