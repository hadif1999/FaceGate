import os
import sys
from loguru import logger
from pathlib import Path
from src.config import ConfigManager


def initialize_logger(parent_dir: str = __file__):
    config = ConfigManager.get_config()
    config_dir = ConfigManager.get_config_dir(False)
    app_dir = config_dir if config_dir is not None else Path(parent_dir).parent.resolve()
    directory = app_dir / "data" / "logs"
    
    # Create directory with proper permissions (rwxr-xr-x)
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o755)
    except PermissionError as e:
        print(f"Failed to create log directory {directory}: {e}")
        raise
    
    # Verify directory is writable
    if not os.access(directory, os.W_OK):
        print(f"Log directory {directory} is not writable")
        raise PermissionError(f"Cannot write to {directory}")
    
    # Define log format
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Configure logger with default extra values
    # logger.configure(extra={"ip": ""})
    
    # Remove all existing handlers
    logger.remove(None)
    
    general_conf = config.general
    
    # Add error log file
    error_log = directory / "error_{time:YYYY-MM-DD}.log"
    try:
        logger.add(error_log, level="ERROR",
                   diagnose=True, backtrace=True,
                   enqueue=True, rotation="50 MB",
                   format=fmt, retention=5)
    except PermissionError as e:
        print(f"Failed to add error logger to {error_log}: {e}")
        raise
    
    # Add general log file based on log level
    log_level: str = config.general.log_level or "INFO"
    general_log = directory / (f"{log_level.lower()}_" + "{time:YYYY-MM-DD}.log")
    try:
        logger.add(general_log, level=log_level,
                   diagnose=True, backtrace=False,
                   enqueue=True, rotation="50 MB",
                   retention=5)
    except PermissionError as e:
        print(f"Failed to add general logger to {general_log}: {e}")
        raise
    
    # Add stdout handler
    logger.add(sys.stdout, level=log_level,
               diagnose=False, backtrace=True, format=fmt)
    
    # Initial logging
    logger.success("app configured")
    logger.info("general configurations: {conf}", conf=config.general)
    logger.info("logs will be saved at {directory}", directory=directory)
