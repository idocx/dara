"""Perform refinements with BGMN."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from dara.bgmn_worker import BGMNWorker
from dara.cif2str import cif2str
from dara.generate_control_file import generate_control_file
from dara.result import RefinementResult, get_result
from dara.xrd import xrdml2xy, raw2xy


def do_refinement(
    pattern_path: Path | str,
    phase_paths: list[Path | str],
    instrument_name: str = "Aeris-fds-Pixcel1d-Medipix3",
    working_dir: Path | str | None = None,
    phase_params: dict | None = None,
    refinement_params: dict | None = None,
    show_progress: bool = False,
) -> RefinementResult:
    """Refine the structure using BGMN."""
    pattern_path = Path(pattern_path)
    phase_paths = [Path(phase_path) for phase_path in phase_paths]
    working_dir = (
        Path(working_dir)
        if working_dir is not None
        else pattern_path.parent / f"refinement_{pattern_path.stem}"
    )

    if not working_dir.exists():
        working_dir.mkdir(exist_ok=True, parents=True)

    if phase_params is None:
        phase_params = {}

    if refinement_params is None:
        refinement_params = {}

    if pattern_path.suffix == ".xrdml":
        pattern_path = xrdml2xy(pattern_path, working_dir)
    elif pattern_path.suffix == ".raw":
        pattern_path = raw2xy(pattern_path, working_dir)

    str_paths = []
    for i, phase_path in enumerate(phase_paths):
        if phase_path.suffix == ".cif":
            str_path = cif2str(phase_path, f"P{i}", working_dir, **phase_params)
        else:
            if phase_path.parent != working_dir:
                shutil.copy(phase_path, working_dir)
            str_path = working_dir / phase_path.name
        str_paths.append(str_path)

    control_file_path = generate_control_file(
        pattern_path,
        str_paths,
        instrument_name,
        working_dir=working_dir,
        **refinement_params,
    )

    bgmn_worker = BGMNWorker()
    bgmn_worker.run_refinement_cmd(control_file_path, show_progress=show_progress)
    return get_result(control_file_path)


def do_refinement_no_saving(
    pattern_path: Path,
    phase_paths: list[Path],
    instrument_name: str = "Aeris-fds-Pixcel1d-Medipix3",
    phase_params: dict | None = None,
    refinement_params: dict | None = None,
    show_progress: bool = False,
) -> RefinementResult:
    """Refine the structure using BGMN in a temporary directory without saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        working_dir = Path(tmpdir)

        return do_refinement(
            pattern_path,
            phase_paths,
            instrument_name,
            working_dir=working_dir,
            phase_params=phase_params,
            refinement_params=refinement_params,
            show_progress=show_progress,
        )
