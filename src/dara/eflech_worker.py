import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd

from dara.generate_control_file import copy_instrument_files, copy_xy_pattern
from dara.xrd import xrdml2xy


class EflechWorker:
    """Functionality for running peak detection using BGMN's eflech and teil executables."""

    def __init__(self):
        self.bgmn_folder = (
            Path(__file__).parent.parent.parent / "bgmn" / "BGMNwin"
        ).absolute()

        self.eflech_path = self.bgmn_folder / "eflech"
        self.teil_path = self.bgmn_folder / "teil"

        if not self.eflech_path.exists():
            raise FileNotFoundError("Cannot find BGMN executable.")

        os.environ["EFLECH"] = self.bgmn_folder.as_posix()
        os.environ["PATH"] += os.pathsep + self.bgmn_folder.as_posix()

    def run_peak_detection(
        self,
        pattern: Union[Path, np.ndarray],
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
                print(pattern_path_temp.read_text())
            else:
                if pattern.suffix == ".xy":
                    copy_xy_pattern(pattern, temp_dir)
                    pattern_path_temp = temp_dir / pattern.name
                elif pattern.suffix == ".xrdml":
                    pattern_path_temp = temp_dir / pattern.with_suffix(".xy").name
                    xrdml2xy(pattern, temp_dir)
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

            peak_list = self.parse_peak_list(temp_dir)

            return peak_list

    @staticmethod
    def generate_control_file(
        pattern_path: Path,
        instrument_name: str,
        *,
        wmin: float = None,
        wmax: float = None,
    ) -> Path:
        control_file_str = f"""
            VERZERR={instrument_name}.geq
            LAMBDA=CU
            % Measured data
            VAL[1]={pattern_path.name}
            {f"WMIN={wmin}" if wmin is not None else ""}
            {f"WMAX={wmax}" if wmax is not None else ""}
            NTHREADS=8
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
            subprocess.run(
                [self.eflech_path.as_posix(), control_file_path.as_posix()],
                cwd=working_dir.as_posix(),
                capture_output=not show_progress,
                timeout=600,
                check=False,
            )
        elif mode == "teil":
            subprocess.run(
                [self.teil_path.as_posix(), control_file_path.as_posix()],
                cwd=working_dir.as_posix(),
                capture_output=not show_progress,
                timeout=600,
                check=False,
            )

    def parse_peak_list(self, par_folder: Path) -> pd.DataFrame:
        all_par_files = list(par_folder.glob("output-*.par"))
        peak_list = []
        for par_file in all_par_files:
            peak_list.extend(self.parse_par_file(par_file))

        peak_list = np.array(peak_list).reshape(-1, 4)

        d_inv = peak_list[:, 0]
        intensity = peak_list[:, 1]
        b1 = peak_list[:, 2]
        b2 = peak_list[:, 3]

        two_theta = np.arcsin(0.15406 * d_inv / 2) * 180 / np.pi * 2

        peak_list_two_theta = np.column_stack((two_theta, intensity, b1, b2))
        peak_list_two_theta = peak_list_two_theta[peak_list_two_theta[:, 0].argsort()]

        df = pd.DataFrame(
            peak_list_two_theta, columns=["2theta", "intensity", "b1", "b2"]
        ).astype(float)

        return df

    @staticmethod
    def parse_par_file(par_file: Path) -> list[list[float]]:
        content = par_file.read_text().split("\n")
        peak_list = []

        if len(content) < 2:
            return peak_list

        peak_num = re.search(r"PEAKZAHL=(\d+)", content[0])

        if not peak_num:
            return peak_list

        peak_num = int(peak_num.group(1))

        if peak_num == 0:
            return peak_list

        for i in range(1, peak_num + 1):
            if i >= len(content):
                break

            numbers = re.split(r"\s+", content[i])

            if numbers:
                rp = int(numbers[0])
                intensity = float(numbers[1])
                d_inv = float(numbers[2])
                if rp == 2:
                    b1 = 0
                    b2 = 0
                elif rp == 3:
                    b1 = float(numbers[3])
                    b2 = 0
                elif rp == 4:
                    b1 = float(numbers[3])
                    b2 = float(numbers[4]) ** 2
                else:
                    b1 = 0
                    b2 = 0

                # Only add peaks with intensity > 0
                if intensity > 0:
                    peak_list.append([d_inv, intensity, b1, b2])

        return peak_list
