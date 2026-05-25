from dataclasses import dataclass
from typing import Literal
import datetime as dt
from pydantic import BaseModel
from uuid import UUID

class QueueMsgSchema(BaseModel):
    uuid: UUID
    msg_type: Literal["RECOGNITION", "REGISTRATION"]
    direction: Literal["incoming", "outgoing", "broadcast"]
    cam_id: int
    face_id: int
    create_date: dt.datetime = dt.datetime.now()
    
    