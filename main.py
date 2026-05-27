import asyncio
import gc
import multiprocessing as mp
import pathlib
import platform
import sys
import time
from typing import List, Optional

import typer
from loguru import logger

from src.config import ConfigManager
from src.Logging import initialize_logger
from src.tasks.recognizer import init_recognizers, start_recognizer_process
from src.tasks.websockets import main_ws_loop

if platform.system() != "Windows":
    import uvloop
else:
    uvloop = None


def _app_dir() -> pathlib.Path:
    if getattr(sys, "frozen", False):
        return pathlib.Path(sys.executable).resolve().parent
    return pathlib.Path(__file__).resolve().parent


def _resolve_config_paths(config_paths: list[str]) -> list[str]:
    resolved_paths = []
    app_dir = _app_dir()

    for config_path in config_paths:
        path = pathlib.Path(config_path)
        if path.exists() or path.is_absolute():
            resolved_paths.append(str(path))
            continue

        exe_neighbor_path = app_dir / path
        if exe_neighbor_path.exists():
            resolved_paths.append(str(exe_neighbor_path))
            continue

        resolved_paths.append(str(path))

    return resolved_paths


def _resolve_runtime_paths():
    config = ConfigManager.get_config()
    config_dir = ConfigManager.get_config_dir(False)
    if config_dir is None:
        return

    data_dir = config_dir / "data"
    logs_dir = data_dir / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    db_path = pathlib.Path(config.vision_setting.face_DB_path)
    if not db_path.is_absolute():
        db_path = (config_dir / db_path).resolve()
        config.vision_setting.face_DB_path = str(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)


def _run_tasks(coro):
    if uvloop is not None:
        return asyncio.run(coro, loop_factory=uvloop.new_event_loop)
    return asyncio.run(coro)


def _configure_multiprocessing():
    mp.freeze_support()
    if platform.system() == "Windows":
        return
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass


async def tasks_runner(interval: float = 0.001, open_camera_window = False):
    config = ConfigManager.get_config()
    config.vision_setting.interval_sec = interval
    recognizer_tasks = init_recognizers(open_camera_window, begin_processes=True)
    websocket_task = asyncio.create_task(main_ws_loop(recognizer_tasks))
    restart_after: dict[int, float] = {}
    
    
    try:
        while True:
            for cam_id, (process, in_queue, out_queue) in list(recognizer_tasks.items()):
                if process.is_alive():
                    continue
                now = time.monotonic()
                if now < restart_after.get(cam_id, 0):
                    continue
                logger.error(f"recognizer process cam_id={cam_id} stopped; restarting")
                try:
                    process.join(timeout=0.1)
                    process.close()
                except Exception:
                    pass
                new_process = start_recognizer_process(cam_id, in_queue, out_queue, None, open_camera_window)
                recognizer_tasks[cam_id] = (new_process, in_queue, out_queue)
                restart_after[cam_id] = now + 5
            await asyncio.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("tasks_runner cancelled")
        for process, _, _ in recognizer_tasks.values():
            process.kill()
    
    except Exception as e:
        logger.exception(f"unhandled exception in tasks_runner: {e}")
        for process, _, _ in recognizer_tasks.values():
            process.kill()
        raise e
    finally:
        websocket_task.cancel()
        await asyncio.gather(websocket_task, return_exceptions=True)
   
    # tasks = [
    #     asyncio.create_task(recognizer_loop(camera_uri=uri, interval=interval,
    #                                         open_camera_window=open_camera_window)),
    #     # add more tasks here
    # ]

    # try:
    #     await asyncio.gather(*tasks)
    # except (asyncio.CancelledError, Exception) as e:
    #     if isinstance(e, asyncio.CancelledError):
    #         logger.info("tasks_runner cancelled")
    #     else:
    #         logger.exception(f"unhandled exception in tasks_runner: {e}")
    # finally:
    #     current = asyncio.current_task()
    #     pending = [t for t in tasks if t is not current and not t.done()]
    #     for t in pending:
    #         t.cancel()
    #     if pending:
    #         await asyncio.gather(*pending, return_exceptions=True)
    #     logger.info("tasks_runner cleanup complete")
        


def CLI(config_paths: List[str] = typer.Option(["config.yaml"], "-c", "--config-path", help="path of config files"),
        initialize_logs: bool = True,
        interval: Optional[float] = typer.Option(None, "-i", "--interval", help="interval of vision"),
        open_camera_window: bool = False):
    
    config = ConfigManager.read_multiple_config_files(*_resolve_config_paths(config_paths))
    config = ConfigManager.get_config()
    _resolve_runtime_paths()
            
    if config is None:
        logger.error("config is not initialized from server, using values from config file")
        return

    if initialize_logs: initialize_logger(__file__)
    
    
    try:
        
        _run_tasks(
            tasks_runner(
                interval=interval or config.vision_setting.interval_sec,
                open_camera_window=open_camera_window or config.general.open_camera_windows,
            )
        )
        
    except SystemExit as e:
        logger.critical(f"exiting app with error_code={e.code} ...")
        raise e
    except Exception as e:
        logger.exception(f"exception occurred in main loop: {e.__class__}: {str(e)}")
    finally:
        gc.collect()

if __name__ == "__main__":
    _configure_multiprocessing()
    typer.run(CLI)
