import asyncio
from loguru import logger
from src.tasks.recognizer import recognizer_loop
from src.config import ConfigManager
import typer
from typing import List
from src.Logging import initialize_logger
import uvloop
import gc, time


async def tasks_runner(camera_idx: int = 0, interval: float = 0.001, open_camera_window = True):
    tasks = [
        asyncio.create_task(recognizer_loop(camera_idx=camera_idx, interval=interval,
                                            open_camera_window=open_camera_window)),
        # add more tasks here
    ]

    try:
        await asyncio.gather(*tasks)
    except (asyncio.CancelledError, Exception) as e:
        if isinstance(e, asyncio.CancelledError):
            logger.info("tasks_runner cancelled")
        else:
            logger.exception(f"unhandled exception in tasks_runner: {e}")
    finally:
        current = asyncio.current_task()
        pending = [t for t in tasks if t is not current and not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        logger.info("tasks_runner cleanup complete")
        


def CLI(config_paths: List[str] = typer.Option(["config.json"], "-c", "--config-path", help="path of config files"),
        initialize_logs: bool = True,
        interval: float = 0.001,
        open_camera_window: bool = True):
    
    config = ConfigManager.read_multiple_config_files(*config_paths)
    config = ConfigManager.get_config()
            
    if config is None:
        logger.error("config is not initialized from server, using values from config file")
        return

    if initialize_logs: initialize_logger(__file__)
    
    
    try:
        
        asyncio.run(tasks_runner(interval=interval, open_camera_window=open_camera_window),
                    loop_factory=uvloop.new_event_loop)
        
    except SystemExit as e:
        logger.critical(f"exiting app with error_code={e.code} ...")
        raise e
    except Exception as e:
        logger.exception(f"exception occurred in main loop: {e.__class__}: {str(e)}")
    finally:
        gc.collect()

if __name__ == "__main__":
    typer.run(CLI)
