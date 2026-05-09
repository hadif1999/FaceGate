
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
    


