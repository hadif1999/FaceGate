
from typing import Coroutine
from loguru import logger 
import asyncio
import uuid
# from tenacity import AsyncRetrying, stop_after_attempt, retry_if_exception_type
from typing import Callable, Awaitable, Tuple, Type
import psutil, os, shlex, sys

import asyncio
from urllib.parse import urlencode, urljoin
from functools import wraps
from typing import TypeVar, Callable, Any, Awaitable, Union, Literal
from inspect import iscoroutinefunction
import time, aiofiles
import hashlib
import numpy as np, cv2
from src.config import Camera

# def restart_app(update_global_config_from_server: bool = False, backend: Literal["mavsdk", "pymavlink"]|None = None):
#     """Restarts the current program, with file objects and descriptors open"""
#     logger.info("Restarting...")
#     python = sys.executable
#     cmd = [python] + sys.argv
#     if not update_global_config_from_server:
#         cmd.append("--no-update-config-from-server")
#     if backend is None:
#         from mavlink_rest.repository import telemetry
#         backend = telemetry.telemetry_backend
#     cmd.append(f"--backend={backend}")
#     logger.debug(f"{cmd = }")
#     os.execv(python, cmd)
    
    
Function = TypeVar('Function', bound=Union[Callable[..., Any], Callable[..., Awaitable[Any]]])

    
def log_exec_time(func: Function) -> Function:
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        result = await func(*args, **kwargs)  # Await the async function
        t2 = time.perf_counter()
        exec_time = t2 - t1
        msg = f"Execution time for {func.__name__}: {exec_time:.4f} seconds"
        logger.debug(msg)
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)  # Call the sync function
        t2 = time.perf_counter()
        exec_time = t2 - t1
        msg = f"Execution time for {func.__name__}: {exec_time:.4f} seconds "
        logger.debug(msg)
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # Return async wrapper for async functions
    else:
        return sync_wrapper  # Return sync wrapper for sync functions
    

def crop_percent(img: np.ndarray, tl_pct: tuple[int, int], br_pct: tuple[int, int]) -> np.ndarray:
    """
    Crop image by moving top-left and bottom-right corners inward by percentages.
    
    Args:
        img: Input image (H x W x C).
        tl_pct: (x%, y%) to shift top-left corner right/down.
        br_pct: (x%, y%) to shift bottom-right corner left/up.
    
    Returns:
        Cropped image copy.
    """
    h, w = img.shape[:2]
    x1 = int(w * tl_pct[0] / 100)
    y1 = int(h * tl_pct[1] / 100)
    x2 = w - int(w * br_pct[0] / 100)
    y2 = h - int(h * br_pct[1] / 100)
    
    # Clamp to valid range
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Empty crop region")
    
    return img[y1:y2, x1:x2].copy()


def shift_point_percent(img: np.ndarray, pt: tuple[int, int], shift_pct: tuple[int, int]) -> tuple[int, int]:
    """
    Shift a point by adding percentages of image width/height, then clamp to image boundaries.
    
    Args:
        img: Input image (provides dimensions).
        pt: (x, y) original point.
        shift_pct: (dx%, dy%) – positive moves right/down, negative moves left/up.
    
    Returns:
        (new_x, new_y) clamped to valid pixel indices.
    """
    h, w = img.shape[:2]
    dx = int(w * shift_pct[0] / 100)
    dy = int(h * shift_pct[1] / 100)
    new_x = pt[0] + dx
    new_y = pt[1] + dy
    
    # Clamp to [0, w-1] and [0, h-1]
    new_x = max(0, min(w - 1, new_x))
    new_y = max(0, min(h - 1, new_y))
    return (new_x, new_y)


def read_frame(cap: cv2.VideoCapture, skip_n_frames: int = 5):
    if cap is None or not cap.isOpened():
        return None

    try:
        for _ in range(max(skip_n_frames, 1)):
            if not cap.grab():
                return None
        ret, frame = cap.retrieve()
        if not ret or frame is None or frame.size == 0:
            return None
        return frame
    except cv2.error as e:
        logger.debug(f"opencv failed while reading frame: {e}")
    except Exception as e:
        logger.debug(f"failed while reading frame: {e}")
    return None



def crop_by_corners(img: np.ndarray, tl_pt: tuple[int, int], br_pt: tuple[int, int]) -> np.ndarray:
    """
    Crop image using top‑left and bottom‑right corner points.
    
    Args:
        img: Input image (H x W x C).
        tl_pt: (x, y) of the top‑left corner.
        br_pt: (x, y) of the bottom‑right corner.
    
    Returns:
        Cropped image copy (region from tl_pt to br_pt).
    """
    h, w = img.shape[:2]
    
    # Unpack points
    x1, y1 = tl_pt
    x2, y2 = br_pt
    
    # Ensure coordinates are in correct order (x1 <= x2, y1 <= y2)
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    
    # Clamp to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Validate non‑empty region
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid crop region: {tl_pt} -> {br_pt} yields empty area")
    
    # Crop (slicing: rows y1..y2, columns x1..x2)
    cropped = img[y1:y2, x1:x2].copy()
    return cropped




def find_cam_idx_by_ip(target_ip: str|None, cams: list[Camera],
                       use_role_if_not_found: bool = True)-> None|int:
    # find by ip
    if target_ip:
        for i, cam in enumerate(cams):
            if target_ip in cam.uri:
                return i
    if not use_role_if_not_found:
        return
    for i, cam in enumerate(cams):
        if cam.can_register:
            return i
    


def restart_app():
    """Restarts the current program, with file objects and descriptors open"""
    logger.info("Restarting...")
    python = sys.executable
    cmd = [python] + sys.argv
    logger.debug(f"{cmd = }")
    os.execv(python, cmd)
