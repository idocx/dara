"""Generate a control file for BGMN."""

from __future__ import annotations

import re
import shutil
import warnings
from pathlib import Path

import numpy as np
from dara.utils import read_phase_name_from_str


def copy_instrument_files(instrument_name: str, working_dir: Path) -> None:
    """Copy the instrument files to the working directory."""
    instrument_path = Path(__file__).parent / "data" / "BGMN-Templates" / "Devices"

    for file in instrument_path.glob(f"{instrument_name}*"):
        shutil.copy(file, working_dir)


def copy_xy_pattern(pattern_path: Path, working_dir: Path) -> Path:
    """Copy the xy pattern to the working directory."""
    # if same directory, do nothing
    if pattern_path.parent != working_dir:
        shutil.copy(pattern_path, working_dir)
    return working_dir / pattern_path.name


def generate_control_file(
    pattern_path: Path,
    str_paths: list[Path],
    instrument_name: str,
    working_dir: Path | None = None,
    *,
    n_threads: int = 8,
    wmin: float | None = None,
    wmax: float | None = None,
    eps2: float | str = "0_-0.01^0.01",
) -> Path:
    """Generate a control file for BGMN."""
    if working_dir is None:
        control_file_path = pattern_path.parent / f"{pattern_path.stem}.sav"
    else:
        control_file_path = working_dir / f"{pattern_path.stem}.sav"

    copy_xy_pattern(pattern_path, control_file_path.parent)
    copy_instrument_files(instrument_name, control_file_path.parent)

    xy_pattern_path = control_file_path.parent / pattern_path.name

    try:
        xy_content = np.loadtxt(pattern_path)
    except ValueError as e:
        raise ValueError(f"Could not load pattern file {pattern_path}")

    if xy_content[:, 1].min() <= 0:
        warnings.warn(
            "Pattern contains negative or zero intensities. Setting them to 1e-6."
        )
        xy_content[:, 1] = np.clip(xy_content[:, 1], 1e-6, None)
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
    LAMBDA=CU
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
    EPS1=0
    {f"PARAM[1]=EPS2={eps2}" if isinstance(eps2, str) else f"EPS2={eps2}"}
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
