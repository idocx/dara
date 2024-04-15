import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd

from dara.bgmn.download_bgmn import download_bgmn
from dara.generate_control_file import copy_instrument_files, copy_xy_pattern
from dara.par_parser import ParParser
from dara.utils import get_logger
from dara.xrd import raw2xy, xrdml2xy

logger = get_logger(__name__)


class EflechWorker:
    """Functionality for running peak detection using BGMN's eflech and teil executables."""

    def __init__(self):
        self.bgmn_folder = (Path(__file__).parent / "bgmn" / "BGMNwin").absolute()

        self.eflech_path = self.bgmn_folder / "eflech"
        self.teil_path = self.bgmn_folder / "teil"

        if (
            not self.eflech_path.exists()
            and not self.eflech_path.with_suffix(".exe").exists()
        ):
            logger.warning("BGMN executable not found. Downloading BGMN.")
            download_bgmn()

        os.environ["EFLECH"] = self.bgmn_folder.as_posix()
        os.environ["PATH"] += os.pathsep + self.bgmn_folder.as_posix()

    def run_peak_detection(
        self,
        pattern: Union[Path, np.ndarray, str],
        instrument_name: str = "Aeris-fds-Pixcel1d-Medipix3",
        show_progress: bool = False,
        *,
        wmin: float = None,
        wmax: float = None,
    ) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            copy_instrument_files(instrument_name, temp_dir)
            if isinstance(pattern, np.ndarray):
                pattern_path_temp = temp_dir / "temp.xy"
                np.savetxt(pattern_path_temp.as_posix(), pattern, fmt="%.6f")
            else:
                if isinstance(pattern, str):
                    pattern = Path(pattern)
                if pattern.suffix == ".xy" or pattern.suffix == ".txt":
                    pattern_path_temp = copy_xy_pattern(pattern, temp_dir)
                elif pattern.suffix == ".xrdml":
                    pattern_path_temp = xrdml2xy(pattern, temp_dir)
                elif pattern.suffix == ".raw":
                    pattern_path_temp = raw2xy(pattern, temp_dir)
                else:
                    raise ValueError(f"Unknown pattern file type: {pattern.suffix}")

            control_file_path = self.generate_control_file(
                pattern_path_temp, instrument_name, wmin=wmin, wmax=wmax
            )

            self.run_eflech(
                control_file_path,
                mode="teil",
                working_dir=temp_dir,
                show_progress=show_progress,
            )
            self.run_eflech(
                control_file_path,
                mode="eflech",
                working_dir=temp_dir,
                show_progress=show_progress,
            )

            peak_list = ParParser(control_file_path).to_df()

            return peak_list

    @staticmethod
    def generate_control_file(
        pattern_path: Path,
        instrument_name: str,
        *,
        wmin: float = None,
        wmax: float = None,
        nthreads: int = None,
    ) -> Path:
        control_file_str = f"""
            VERZERR={instrument_name}.geq
            LAMBDA=CU
            % Measured data
            VAL[1]={pattern_path.name}
            {f"WMIN={wmin}" if wmin is not None else ""}
            {f"WMAX={wmax}" if wmax is not None else ""}
            NTHREADS={nthreads if nthreads is not None else os.cpu_count()}
            TEST=ND234U
            OUTPUTMASK=output-$
            TITELMASK=output-$"""

        control_file_str = "\n".join(
            [line.strip() for line in control_file_str.split("\n")]
        )
        control_file_path = pattern_path.parent / "control.sav"

        with control_file_path.open("w") as f:
            f.write(control_file_str)

        return control_file_path

    def run_eflech(
        self,
        control_file_path: Path,
        mode: Literal["eflech", "teil"],
        working_dir: Path,
        show_progress: bool = False,
    ):
        if mode == "eflech":
            cp = subprocess.run(
                [self.eflech_path.as_posix(), control_file_path.as_posix()],
                cwd=working_dir.as_posix(),
                capture_output=not show_progress,
                timeout=1200,
                check=False,
            )
        elif mode == "teil":
            cp = subprocess.run(
                [self.teil_path.as_posix(), control_file_path.as_posix()],
                cwd=working_dir.as_posix(),
                capture_output=not show_progress,
                timeout=1200,
                check=False,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if cp.returncode:
            raise RuntimeError(
                f"Error in BGMN {mode} for {control_file_path}. The exit code is {cp.returncode}\n"
                f"{cp.stdout}\n"
                f"{cp.stderr}"
            )
