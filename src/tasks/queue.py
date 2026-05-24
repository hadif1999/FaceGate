from dataclasses import dataclass
from typing import Literal
import datetime as dt
from pydantic import BaseModel


class QueueMsgSchema(BaseModel):
    msg_type: Literal["RECOGNITION", "REGISTRATION"]
    cam_id: int
    face_id: int
    create_date: dt.datetime = dt.datetime.now()
    
    