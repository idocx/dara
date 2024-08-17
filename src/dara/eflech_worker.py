from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from dara.bgmn.download_bgmn import download_bgmn
from dara.generate_control_file import (
    copy_instrument_files,
    copy_xy_pattern,
    trim_pattern,
)
from dara.utils import get_logger, get_wavelength, intensity_correction
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
        pattern: Path | np.ndarray | str,
        wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float = "Cu",
        instrument_name: str = "Aeris-fds-Pixcel1d-Medipix3",
        show_progress: bool = False,
        *,
        wmin: float = None,
        wmax: float = None,
        nthreads: int = None,
        timeout: int = 1800,
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
                if (
                    pattern.suffix == ".xy"
                    or pattern.suffix == ".txt"
                    or pattern.suffix == ".xye"
                ):
                    pattern_path_temp = copy_xy_pattern(pattern, temp_dir)
                elif pattern.suffix == ".xrdml":
                    pattern_path_temp = xrdml2xy(pattern, temp_dir)
                elif pattern.suffix == ".raw":
                    pattern_path_temp = raw2xy(pattern, temp_dir)
                else:
                    raise ValueError(f"Unknown pattern file type: {pattern.suffix}")

            xy_content = np.loadtxt(pattern_path_temp, dtype=float)
            xy_content = trim_pattern(xy_content)
            np.savetxt(pattern_path_temp, xy_content, fmt="%.6f")

            control_file_path = self.generate_control_file(
                pattern_path_temp,
                wavelength=wavelength,
                instrument_name=instrument_name,
                wmin=wmin,
                wmax=wmax,
                nthreads=nthreads,
            )

            teil_output = self.run_eflech(
                control_file_path,
                mode="teil",
                working_dir=temp_dir,
                show_progress=show_progress,
                timeout=timeout,
            )

            self.run_eflech(
                control_file_path,
                mode="eflech",
                working_dir=temp_dir,
                show_progress=show_progress,
                timeout=timeout,
            )

            return self.parse_peak_list(temp_dir, wavelength=wavelength)

    @staticmethod
    def generate_control_file(
        pattern_path: Path,
        wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float,
        instrument_name: str,
        *,
        wmin: float = None,
        wmax: float = None,
        nthreads: int = None,
    ) -> Path:
        control_file_str = f"""
            VERZERR={instrument_name}.geq
            {f"LAMBDA={wavelength.upper()}" if isinstance(wavelength, str) else f"SYNCHROTRON={wavelength:.4f}"}
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
        timeout: int = 1800,
    ) -> str:
        if mode == "eflech":
            cp = subprocess.run(
                [self.eflech_path.as_posix(), control_file_path.as_posix()],
                cwd=working_dir.as_posix(),
                capture_output=not show_progress,
                timeout=timeout,
                check=False,
            )
        elif mode == "teil":
            cp = subprocess.run(
                [self.teil_path.as_posix(), control_file_path.as_posix()],
                cwd=working_dir.as_posix(),
                capture_output=not show_progress,
                timeout=timeout,
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
        return cp.stdout.decode()

    def parse_peak_list(
        self,
        par_folder: Path,
        wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float,
    ) -> pd.DataFrame:
        all_par_files = list(par_folder.glob("output-*.par"))
        peak_list = []
        wavelength_float = get_wavelength(wavelength)
        for par_file in all_par_files:
            peak_list.extend(self.parse_par_file(par_file, wavelength=wavelength_float))

        peak_list = np.array(peak_list).reshape(-1, 4)

        d_inv = peak_list[:, 0]
        intensity = peak_list[:, 1]
        b1 = peak_list[:, 2]
        b2 = peak_list[:, 3]

        two_theta = np.arcsin(wavelength_float * d_inv / 2) * 180 / np.pi * 2

        peak_list_two_theta = np.column_stack((two_theta, intensity, b1, b2))
        peak_list_two_theta = peak_list_two_theta[peak_list_two_theta[:, 0].argsort()]

        return pd.DataFrame(
            peak_list_two_theta, columns=["2theta", "intensity", "b1", "b2"]
        ).astype(float)

    @staticmethod
    def parse_par_file(par_file: Path, wavelength: float) -> list[list[float]]:
        content = par_file.read_text().split("\n")
        peak_list = []

        if len(content) < 2:
            return peak_list

        peak_num = re.search(r"PEAKZAHL=(\d+)", content[0])
        pol = re.search(r"POL=(\d+(\.\d+)?)", content[0])
        pol = float(pol.group(1)) if pol else 1.0

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
                gsum = (
                    1.0
                    if (gsum := re.search("GSUM=(\\d+(\\.\\d+)?)", content[i])) is None
                    else float(gsum.group(1))
                )
                intensity = intensity_correction(
                    intensity=intensity,
                    d_inv=d_inv,
                    gsum=gsum,
                    wavelength=wavelength,
                    pol=pol,
                )

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
