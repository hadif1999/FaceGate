from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Literal
from loguru import logger
import os
import aiofiles
import json
from typing import Literal

   

class DetectionSetting(BaseModel):
    conf_thresh: float = 0.75
    nms_thresh: float = 0.6
    top_k:int = 10
    model_name: Literal["yunet"] = "yunet"

    @field_validator("conf_thresh", "nms_thresh")
    @classmethod
    def threshold_between_zero_and_one(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("must be between 0 and 1")
        return value


class RecognitionSetting(BaseModel):
    conf_thresh: float = 0.75
    similarity_func: Literal["cosine", "l2"] = "cosine"
    model_name: Literal["sface"] = "sface"
    after_recognition_delay: int = 3
    

class CropSetting(BaseModel):
    tl_reduce_pct: list[int] = Field(max_length=2, min_length=2)
    br_reduce_pct: list[int] = Field(max_length=2, min_length=2)


class VisionSetting(BaseModel):
    face_DB_path: str
    models_path: str
    skip_n_frames: int = 5
    interval_sec: float = 0.01
    crop: CropSetting
    detection: DetectionSetting = Field(default_factory=DetectionSetting)
    recognition: RecognitionSetting = Field(default_factory=RecognitionSetting)


class Camera(BaseModel):
    uri: str = Field(alias="URI")
    can_register: bool = False

    model_config = ConfigDict(populate_by_name=True)

    
class RestAPI(BaseModel):
    port: int = 10821
    host: str = "0.0.0.0"
    global_prefix: str = "/api/v1"
    global_timeout: int = 20
    as_https: bool = False
    ssl_keyfile_dir: str|None = None
    ssl_certfile_dir: str|None = None 



class General(BaseModel):
    log_level: str = "DEBUG"
    open_camera_windows: bool = False
    

    
class HealthCheck(BaseModel):
    enabled: bool = True
    push_to_server: bool = True
    route: str
    interval_sec: int = 5


class WebsocketServer(BaseModel):
    url: str
    
    
    


# Main config class
class AppConfig(BaseSettings):
    general: General
    cameras: list[Camera] = Field(default_factory=list)
    vision_setting: VisionSetting = Field(default_factory=VisionSetting)
    rest_api: RestAPI
    websocket_server: WebsocketServer
    health_check: HealthCheck
    model_config = SettingsConfigDict(
        case_sensitive=False,           # Environment variables are case-insensitive
        env_prefix="LATIKA__",         # Prefix for all env vars (e.g., MAVLINK__GENERAL__LOG_LEVEL)
        env_file=".env",                # Load from .env file if present
        env_nested_delimiter="__",      # Delimiter for nested fields
        extra="ignore"                  # Ignore extra fields not defined in the model
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        # Only use environment variables and .env file if is_production is False
        config = init_settings.init_kwargs
        if config and isinstance(config, dict) and 'general' in config:
            is_production = config['general'].get('is_production', True)
            if not is_production:
                return (init_settings, env_settings, dotenv_settings, file_secret_settings)
        return (init_settings,)
    
    
class ConfigManager:
    __CONFIG: Optional[AppConfig] = None
    __CONFIG_PATH: str|None = None
    
    
    @staticmethod
    def read_config_file(path:str, make_IfNotExist: bool = False, strict: bool = True) -> AppConfig:
        """read json file data

        Args:
            path (str): _description_
            make_IfNotExist (bool, optional): _description_. Defaults to False.

        Returns:
            AppConfig: _description_
        """        
        import pathlib
        _path = pathlib.Path(path)
        if not _path.exists() and make_IfNotExist: # creating if not exists
            with open(_path, 'a') as file: 
                os.makedirs(_path.parent, exist_ok=True)
                _config = AppConfig()
                data: str = _config.model_dump_json(indent=4, exclude_none=True)
                file.write(data)
                logger.debug(f"making new config file with {path=}")
                __class__.__CONFIG = _config
                __class__.__CONFIG_PATH = path
                return _config
        else: 
            with open(_path, 'r') as file:
                logger.debug(f"reading existing config file with {path=}")
                if strict:
                    _config = AppConfig.model_validate_json(file.read())
                else:
                    _config = AppConfig.model_construct(file.read())
                __class__.__CONFIG = _config
                __class__.__CONFIG_PATH = path
                return _config
            
        
    
    @staticmethod
    def read_multiple_config_files(*paths) -> AppConfig:
        """
        Reads multiple config files and merges them into a single AppConfig object.
        If a file does not exist, it can create a new one if make_IfNotExist is True.
        """
        import importlib
        from pydantic import ValidationError
        merged_config_raw = {}
        for i, path in enumerate(paths):
            if ".json" in path:
                with open(path, 'r') as file:
                    config_raw: dict = json.loads(file.read())
                merged_config_raw.update(config_raw)
                __class__.__CONFIG_PATH = paths[i] # set json conf as public file
            elif ".yaml" in path:
                import yaml
                with open(path, "r") as file:
                    config_raw: dict = yaml.safe_load(file)
                merged_config_raw.update(config_raw)
            elif ".py" in path:
                config_raw: dict = importlib.import_module(path.removesuffix(".py")).config
                merged_config_raw.update(config_raw)
            else:
                logger.error(msg:=f"non-defined config type : {path}")
                raise ValueError(msg)     
        try:           
            config = AppConfig.model_validate(merged_config_raw)  # Validate the merged config
            __class__.__CONFIG = config                
        except ValidationError as e:
            logger.error(f"exception thrown in config validation, maybe all required fields hasn't passed yet: \n {e}")
            raise e
        return config
 
    
    # @staticmethod
    # async def async_overwrite_config(config: AppConfig):        
    #     ConfigManager.update_config(config)
    #     async with aiofiles.open(ConfigManager.get_config_path(), 'w') as file:
    #         await file.write(config.model_dump_json())
    #     return config
    
 
    @staticmethod
    async def async_read_config_file(path:str, make_IfNotExist: bool = False) -> AppConfig:
        import pathlib
        _path = pathlib.Path(path)
        if not _path.exists() and make_IfNotExist: # creating if not exists
            async with aiofiles.open(_path, 'a') as file: 
                os.makedirs(_path.parent, exist_ok=True)
                _config = AppConfig()
                data: str = _config.model_dump_json(indent=4)
                await file.write(data)
                logger.debug(f"making new config file with {path=}")
                __class__.__CONFIG = _config
                __class__.__CONFIG_PATH = path
                return _config
        else: 
            async with aiofiles.open(_path, 'r') as file:
                logger.debug(f"reading existing config file with {path=}")
                _config = AppConfig.model_validate_json(await file.read())
                __class__.__CONFIG = _config
                __class__.__CONFIG_PATH = path
                return _config
            
        
    @staticmethod
    async def async_overwrite_config_file(new_config: AppConfig|dict):
        new_conf = AppConfig.model_validate(new_config)
        if new_conf in [None, {}]:
            logger.error("validated config is invalid")
            return 
        __class__.update_config(new_conf)
        path = __class__.get_config_path()
        async with aiofiles.open(path, 'r') as _file:
            config_pub: dict = json.loads(await _file.read())
        if None in [config_pub, path] or config_pub == {}:
            logger.error("path or old_config is not valid (None or {})")
            return 
        # remove field of main config not found in public conf
        new_conf_filtered = new_conf.model_dump(exclude_none=True).copy()
        conf_to_save = {key: val for key, val in new_conf_filtered.items() if key in config_pub.keys()}
        if None in [new_conf_filtered, conf_to_save] or {} in [new_conf_filtered]:
            logger.error("new_conf_filtered or conf_to_save is not valid (None or {})")
            return 
        #### 
        async with aiofiles.open(path, 'w') as _file:
            await _file.write(json.dumps(conf_to_save, indent=4))
        return new_conf
    
    
    def overwrite_config_file(new_config: AppConfig|dict):
        new_conf = AppConfig.model_validate(new_config)
        __class__.update_config(new_conf)
        path = __class__.get_config_path()
        with open(path, 'w') as _file:
            _file.write(new_conf.model_dump_json(indent=4, exclude_unset=True))
        return new_conf
    
        
            
    @staticmethod
    def get_config(raise_ifNone: bool = True) -> AppConfig:
        config = __class__.__CONFIG
        if config == None and raise_ifNone:
            raise ValueError("CONFIG object is None")
        return config
    
    
    @staticmethod
    def get_config_path(raise_ifNone: bool = True) -> str:
        path = __class__.__CONFIG_PATH
        if path is None and raise_ifNone:
            raise ValueError("config path is None")
        return path
            
            
    @classmethod
    def update_config(cls, config: AppConfig) -> AppConfig:
        cls.__CONFIG = config
        return config
            
