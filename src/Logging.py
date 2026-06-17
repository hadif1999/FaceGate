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

    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o755)
    except PermissionError as e:
        print(f"Failed to create log directory {directory}: {e}")
        raise

    if not os.access(directory, os.W_OK):
        print(f"Log directory {directory} is not writable")
        raise PermissionError(f"Cannot write to {directory}")

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Remove every existing sink so the selected log level takes effect exactly once.
    logger.remove()

    def _resolve_log_level(value: str | None) -> str:
        level = (value or "INFO").strip().upper()
        try:
            logger.level(level)
        except ValueError:
            level = "INFO"
        return level

    log_level = _resolve_log_level(config.general.log_level)

    error_log = directory / "error_{time:YYYY-MM-DD}.log"
    try:
        logger.add(
            error_log,
            level="ERROR",
            diagnose=True,
            backtrace=True,
            enqueue=True,
            rotation="50 MB",
            format=fmt,
            retention=5,
        )
    except PermissionError as e:
        print(f"Failed to add error logger to {error_log}: {e}")
        raise

    general_log = directory / (f"{log_level.lower()}_" + "{time:YYYY-MM-DD}.log")
    try:
        logger.add(
            general_log,
            level=log_level,
            diagnose=True,
            backtrace=False,
            enqueue=True,
            rotation="50 MB",
            format=fmt,
            retention=5,
        )
    except PermissionError as e:
        print(f"Failed to add general logger to {general_log}: {e}")
        raise

    logger.add(sys.stdout, level=log_level, diagnose=False, backtrace=True, format=fmt)

    logger.success("app configured")
    logger.info("general configurations: {conf}", conf=config.general)
    logger.info("logs will be saved at {directory}", directory=directory)
