"""Default DARA settings. This approach was inspired by the atomate2 package."""

import warnings
from pathlib import Path
from typing import Optional

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

    RAY_NUM_CPUS: Optional[int] = Field(
        None,
        description=(
            "Number of CPUs to give Ray's own resource accounting "
            "(passed as `num_cpus=` to `ray.init()`). If unset, auto-detected "
            "from an explicit SLURM allocation, then OS-level CPU affinity "
            "(respects cgroup/container restrictions), then the total CPU "
            "count -- see `dara.resource_detection.detect_available_cores`."
        ),
    )
    BGMN_N_THREADS: Optional[int] = Field(
        None,
        description=(
            "Threads per BGMN refinement task (the `n_threads` refinement "
            "param). If unset, auto-derived from the detected core count -- "
            "see `dara.resource_detection.default_bgmn_n_threads`. BGMN's "
            "own thread scaling saturates early (measured: 1.36x speedup at "
            "2 threads vs. 1.71x at 10, on a 12-core machine), so the "
            "default favors more concurrent Ray tasks (wider tree-search "
            "exploration) over one heavily-threaded task."
        ),
    )

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
