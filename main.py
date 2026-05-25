import asyncio
from loguru import logger
from src.tasks.recognizer import recognizer_loop
from src.config import ConfigManager
import typer
from typing import List, Optional
from src.Logging import initialize_logger
import uvloop
import gc, time
from src.tasks.recognizer import init_recognizers
import multiprocessing as mp

mp.set_start_method("fork")


async def tasks_runner(interval: float = 0.001, open_camera_window = False):
    config = ConfigManager.get_config()
    config.vision_setting.interval_sec = interval
    recognizer_tasks = init_recognizers(open_camera_window, begin_processes=True)
    
    
    try:
        while True:
            all_tasks_healthy = all([task.is_alive() for task in recognizer_tasks])
            if not all_tasks_healthy:
                break
            await asyncio.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("tasks_runner cancelled")
        for task in recognizer_tasks:
            task.kill()
    
    except Exception as e:
        logger.exception(f"unhandled exception in tasks_runner: {e}")
        for task in recognizer_tasks:
            task.kill()
        raise e
   
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
    
    config = ConfigManager.read_multiple_config_files(*config_paths)
    config = ConfigManager.get_config()
            
    if config is None:
        logger.error("config is not initialized from server, using values from config file")
        return

    if initialize_logs: initialize_logger(__file__)
    
    
    try:
        
        asyncio.run(tasks_runner(interval=interval or config.vision_setting.interval_sec,
                                open_camera_window=open_camera_window or config.general.open_camera_windows),
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
