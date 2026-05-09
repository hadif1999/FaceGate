import uvicorn
import typer, asyncio
import time
import gc
from loguru import logger
from typing import List, Literal
import uvloop
from enum import Enum
from src.config import ConfigManager
from src.Logging import initialize_logger

    
    
async def run_main_tasks(server_conf: uvicorn.Config,
                         verbose: bool = True):
    config = ConfigManager.get_config()
    
    
    async def monitor_app(interval: int = 5):
        while True:
            await asyncio.sleep(interval)    
    
    if verbose:
        tasks.append(asyncio.create_task(monitor_app(interval=5), name="TaskMonitor"))
        
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Main tasks cancelled")
    except SystemExit as e:
        logger.warning(f"SystemExit triggered: {e}")
        raise e
    except Exception as e:
        logger.exception(f"Exception occurred in main tasks: {e}")
    finally:
        # Graceful cleanup
        logger.info("Cleaning up tasks...")
        current_task = asyncio.current_task()
        for task in tasks:
            if task is not current_task and not task.done():
                task.cancel()
        
        # return_exceptions=True is key to preventing crashes during cleanup
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Give MAVSDK/gRPC threads time to release handles before loop closes
        await asyncio.sleep(0.5)





def CLI(config_paths: List[str] = typer.Option(["config.json"], "-c", "--config-path", help="path of config files"),
        port: int|None = None,
        initialize_logs: bool = True,
        verbose: bool = True):
    
    config = ConfigManager.read_multiple_config_files(*config_paths)
    config = ConfigManager.get_config()
            
    if config is None:
        logger.error("config is not initialized from server, using values from config file")
        return

    if initialize_logs: initialize_logger(__file__)
    
    
    rest_conf = config.rest_api
    _port, _host = rest_conf.port, rest_conf.host
    is_ssl = rest_conf.as_https
    ssl_key_dir, ssl_cert_dir = (rest_conf.ssl_keyfile_dir, rest_conf.ssl_certfile_dir) if is_ssl else (None, None)
    # uvicorn_config = uvicorn.Config(app, port=port or _port, host=_host, # toDo
    #                  ssl_keyfile=ssl_key_dir, ssl_certfile=ssl_cert_dir)
    
    while True:
        try:
            
            asyncio.run(run_main_tasks(uvicorn_config, verbose=verbose),
                        loop_factory=uvloop.new_event_loop)
            
        except SystemExit as e:
            logger.critical(f"exiting app with error_code={e.code} ...")
            raise e
        except Exception as e:
            logger.exception(f"exception occurred in main loop: {e.__class__}: {str(e)}")
            time.sleep(1)
            continue
        finally:
            # FIX 3: Force Garbage Collection
            gc.collect()

if __name__ == "__main__":
    typer.run(CLI)