def load_app_config():
    """Loads the application configuration."""
    try:
        with open("./config/app_config.toml", "rb") as f:
            config = tomllib.load(f)
        config["camera"]["common"]["interval"] = 1.0 / config["camera"]["common"]["fps"]
        return config
    except FileNotFoundError:
        raise FileNotFoundError("Configuration file not found.")
    except tomllib.TOMLDecodeError:
        raise ValueError("Error decoding configuration file.")