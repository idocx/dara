"""Generate a control file for BGMN."""

from __future__ import annotations

import re
import shutil
import warnings
from pathlib import Path
from typing import Literal

import numpy as np

from dara.settings import DaraSettings
from dara.utils import read_phase_name_from_str

_DARA_SETTINGS = DaraSettings()


def copy_instrument_files(instrument_profile: str | Path, working_dir: Path) -> str:
    """
    Copy the instrument file (.geq) to the working directory.

    Args:
        working_dir: the working directory

    Returns
    -------
        The name of the instrument
    """
    default_instrument_path = (
        Path(__file__).parent / "data" / "BGMN-Templates" / "Devices"
    )
    instrument_path = Path(instrument_profile)  # try to parse as a path
    if instrument_path.suffix != ".geq" or not instrument_path.exists():
        instrument_profile = instrument_path.name.removesuffix(".geq")
        instrument_path = default_instrument_path / f"{instrument_profile}.geq"

    if not instrument_path.exists():
        raise FileNotFoundError(
            f"Could not find the instrument file ({instrument_profile} in both "
            f"the provided path and the default path ({default_instrument_path})."
        )

    shutil.copy(instrument_path, working_dir)
    return instrument_path.stem


def copy_xy_pattern(pattern_path: Path, working_dir: Path) -> Path:
    """Copy the xy pattern to the working directory."""
    # if same directory, do nothing
    if pattern_path.parent != working_dir:
        shutil.copy(pattern_path, working_dir)
    return working_dir / pattern_path.name


def trim_pattern(xy_content: np.ndarray) -> np.ndarray:
    """Trim the pattern to remove negative intensities."""
    if xy_content[:, 1].min() <= 0:
        warnings.warn(
            "Pattern contains negative or zero intensities. Setting them to 1e-6."
        )
        xy_content[:, 1] = np.clip(xy_content[:, 1], 1e-6, None)

    if xy_content[:, 0].min() < 1.0:
        warnings.warn("Pattern contains 2-theta values below 1.0. Remove them.")
        xy_content = xy_content[xy_content[:, 0] >= 1.0]

    return xy_content


def generate_control_file(
    pattern_path: Path,
    str_paths: list[Path],
    instrument_profile: str | Path,
    working_dir: Path | None = None,
    *,
    n_threads: int = _DARA_SETTINGS.BGMN_N_THREADS,
    wmin: float | None = None,
    wmax: float | None = None,
    eps1: float | str = 0.0,
    eps2: float | str = "0_-0.05^0.05",
    wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float = "Cu",
) -> Path:
    """
    Generate a control file for BGMN.

    Args:
        pattern_path: the path to the pattern file. It has to be in `.xy` format
        str_paths: the paths to the STR files
        instrument_profile: the name of the instrument, if it is a path, it must be ended with `.geq`
        working_dir: the working directory
        n_threads: the number of threads to use
        wmin: the minimum wavelength
        wmax: the maximum wavelength
        eps1: the epsilon1 value, it is used to refine zero point
        eps2: the epsilon2 value, it is used to refine sample height
        wavelength: the wavelength to use. If a float is provided, it is used as the
            wavelength in nm (synchrotron radiation). If a string is provided, it is
            the target material in X-ray tubes.

    """
    if working_dir is None:
        control_file_path = pattern_path.parent / f"{pattern_path.stem}.sav"
    else:
        control_file_path = working_dir / f"{pattern_path.stem}.sav"

    copy_xy_pattern(pattern_path, control_file_path.parent)
    instrument_name = copy_instrument_files(
        instrument_profile, control_file_path.parent
    )

    xy_pattern_path = control_file_path.parent / pattern_path.name

    try:
        xy_content = np.loadtxt(pattern_path)
    except ValueError as e:
        raise ValueError(f"Could not load pattern file {pattern_path}") from e

    xy_content = trim_pattern(xy_content)
    np.savetxt(xy_pattern_path, xy_content, fmt="%.6f")

    phases_str = "\n".join(
        [f"STRUC[{i}]={str_path.name}" for i, str_path in enumerate(str_paths, start=1)]
    )

    phase_names = [read_phase_name_from_str(str_path) for str_path in str_paths]
    phase_fraction_str = "\n".join(
        [f"Q{phase_name}={phase_name}/sum" for phase_name in phase_names]
    )
    goal_str = "\n".join(
        [
            f"GOAL[{i}]=Q{phase_name}"
            for i, phase_name in enumerate(phase_names, start=1)
        ]
    )

    control_file = f"""
    % Theoretical instrumental function
    VERZERR={instrument_name}.geq
    % Wavelength
    {f"LAMBDA={wavelength.upper()}" if isinstance(wavelength, str) else f"SYNCHROTRON={wavelength:.4f}"}
    {f"WMIN={wmin}" if wmin is not None else ""}
    {f"WMAX={wmax}" if wmax is not None else ""}
    % Phases
    {phases_str}
    % Measured data
    VAL[1]={pattern_path.name}
    % Result list output
    LIST={pattern_path.stem}.lst
    % Peak list output
    OUTPUT={pattern_path.stem}.par
    % Diagram output
    DIAGRAMM={pattern_path.stem}.dia
    % Global parameters for zero point and sample displacement
    {f"PARAM[1]=EPS1={eps1}" if isinstance(eps1, str) else f"EPS1={eps1}"}
    {f"PARAM[{'2' if isinstance(eps1, str) else '1'}]=EPS2={eps2}" if isinstance(eps2, str) else f"EPS2={eps2}"}
    NTHREADS={n_threads}
    PROTOKOLL=Y
    sum={"+".join(phase_name for phase_name in phase_names)}
    {phase_fraction_str}
    {goal_str}
    """
    control_file = re.sub(r"^\s+", "", control_file, flags=re.MULTILINE)

    with open(control_file_path, "w") as f:
        f.write(control_file)

    return control_file_path
