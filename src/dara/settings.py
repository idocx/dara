"""Default DARA settings. This approach was inspired by the atomate2 package."""

import warnings
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_CONFIG_FILE_PATH = "~/.dara.yaml"


class DaraSettings(BaseSettings):
    """
    Settings for DARA.

    To edit these, please modify the config YAML file. By default, this is located at
    ~/.dara.yaml, but one can specify the path by setting the environment variable:
    "DARA_CONFIG_FILE".
    """

    CONFIG_FILE: str = Field(_DEFAULT_CONFIG_FILE_PATH, description="File to load alternative defaults from.")

    PATH_TO_ICSD: Path = Field(Path("~/ICSD_2024/ICSD_2024_experimental_inorganic/experimental_inorganic").expanduser())
    PATH_TO_COD: Path = Field(Path("~/COD_2024").expanduser())

    model_config = SettingsConfigDict(env_prefix="dara_")  # prepend dara_ to env vars

    @model_validator(mode="before")
    @classmethod
    def load_default_settings(cls, values) -> dict:
        """
        Load settings from file or environment variables.

        Loads settings from a root file if available and uses that as defaults in
        place of built-in defaults.

        This allows setting of the config file path through environment variables.
        """
        from monty.serialization import loadfn

        config_file_path = values.get("CONFIG_FILE", _DEFAULT_CONFIG_FILE_PATH)
        config_file_path = Path(config_file_path).expanduser()

        new_values = {}
        if config_file_path.exists():
            if config_file_path.stat().st_size == 0:
                warnings.warn(
                    f"DARA config file is empty: {config_file_path}",
                    stacklevel=2,
                )
            else:
                try:
                    new_values.update(loadfn(config_file_path))
                except ValueError:
                    raise SyntaxError(f"DARA config file is unparsable:{config_file_path} ") from None

        return {**new_values, **values}
