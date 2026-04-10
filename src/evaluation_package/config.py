import platform
import warnings
from pathlib import Path
from ruamel.yaml import YAML

yaml = YAML(typ='safe', pure=True)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

class Config:
    def __init__(self):
        self._cfg = self.load_config()

    def load_config(self) -> dict:
        if not CONFIG_PATH.exists():
            warnings.warn(f"Config file not found at {CONFIG_PATH}. Using empty configuration.")
            return {}
        
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.load(f)
            if cfg is None:
                cfg = {}
            return cfg

    @property
    def data_folder_home(self) -> Path:
        sys_name = platform.system()
        paths = self._cfg.get("data_folder_home", {})
        if sys_name in paths:
            return Path(paths[sys_name])
        elif "default" in paths:
            return Path(paths["default"])
        else:
            warnings.warn(f"No configured data_folder_home for OS '{sys_name}', falling back to current working directory.")
            return Path.cwd()

    @property
    def rf_calibration_device_key(self) -> str:
        return self._cfg.get("rf_calibration_device_key", "rf_source")

config = Config()
