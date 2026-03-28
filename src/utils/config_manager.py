class DotDict(dict):
    """A dictionary that supports dot notation for accessing keys."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


class ConfigManager:
    def __init__(self, config):
        self.config = DotDict(config)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

    def get_config(self):
        return self.config
