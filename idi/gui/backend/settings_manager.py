import json
import os
from typing import Dict, Any

class SettingsManager:
    def __init__(self, config_path="idi/gui/settings.json"):
        self.config_path = os.path.abspath(config_path)
        self.defaults = {
            "market_sim": {
                "volatility": 0.01,
                "drift_bull": 0.002,
                "drift_bear": -0.002,
                "fee_bps": 5.0
            },
            "ui": {
                "theme": "dark",
                "animations": True
            }
        }
        self.settings = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            self._save(self.defaults)
            return self.defaults
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception:
            return self.defaults

    def _save(self, settings: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(settings, f, indent=2)

    def get_settings(self) -> Dict[str, Any]:
        return self.settings

    def update_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in new_settings.items():
            if isinstance(v, dict) and k in self.settings and isinstance(self.settings[k], dict):
                self.settings[k].update(v)
            else:
                self.settings[k] = v
        self._save(self.settings)
        return self.settings
