"""Perform refinements with BGMN."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dara.bgmn_worker import BGMNWorker
from dara.cif2str import cif2str
from dara.generate_control_file import generate_control_file
from dara.result import RefinementMetrics, RefinementResult, get_result
from dara.xrd import convert_pattern_to_xy


class RefinementPhase(BaseModel, frozen=True):
    """
    Input phase for refinement.

    Contains the path to the phase file and the specific parameters for the phase.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    path: Path = Field(..., description="The path to the phase file.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        kw_only=True,
        description="The specific parameters for the phase.",
    )

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, v):
        return Path(v)

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other: RefinementPhase):
        return self.path == other.path

    @classmethod
    def make(cls, path_obj: RefinementPhase | Path | str) -> RefinementPhase:
        """
        Make an RefinementPhase object from a path object. If the path object is already an
        RefinementPhase object, return it.
        If the path object is a string or Path object, create an RefinementPhase object
        with the path object with no specific parameters (the default parameters will be used).

        Args:
            path_obj: the path object, can be a string, Path object, or RefinementPhase object.

        Returns
        -------
            RefinementPhase object
        """
        return (
            path_obj
            if isinstance(path_obj, RefinementPhase)
            else RefinementPhase(path=Path(path_obj))
        )


def _attach_peak_markers(
    result: RefinementResult,
    pattern_path: Path,
    wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float,
    instrument_profile: str | Path,
    use_residual: bool = True,
    residual_integral_fraction: float = 0.010,
    residual_calc_coverage_ratio: float = 0.35,
    residual_window_detect_fraction: float = 0.003,
    missing_intensity_ratio: float = 0.005,
    extra_intensity_ratio: float = 0.03,
    intensity_mismatch_height_tolerance: float = 0.40,
    intensity_mismatch_area_tolerance: float = 0.60,
) -> None:
    """Run the full peak-matching pipeline and store results on *result* in-place.

    Detects observed peaks, matches them against the refined calc pattern, and
    stores the resulting missing/extra/intensity-mismatch markers (plus rwp) on
    ``result.refinement_metrics``.

    Args:
        use_residual: whether to additionally flag broad unfit regions via the
            integrated-residual scan (see ``find_residual_regions``).
        residual_integral_fraction: passed through to ``find_residual_regions``
            as ``integral_fraction``.
        residual_calc_coverage_ratio: passed through to ``find_residual_regions``
            as ``calc_coverage_ratio``.
        residual_window_detect_fraction: passed through to
            ``find_residual_regions`` as ``window_detect_fraction``.
        missing_intensity_ratio: minimum intensity ratio for isolated missing
            peaks (see ``PeakMatcher.get_isolated_peaks``).
        extra_intensity_ratio: minimum intensity ratio for isolated extra peaks
            (see ``PeakMatcher.get_isolated_peaks``).
        intensity_mismatch_height_tolerance: passed through to
            ``find_intensity_mismatch_peaks`` as ``height_tolerance``.
        intensity_mismatch_area_tolerance: passed through to
            ``find_intensity_mismatch_peaks`` as ``area_tolerance``.
    """
    from dara.peak_detection import detect_peaks
    from dara.search.peak_matcher import (
        PeakMatcher,
        find_intensity_mismatch_peaks,
        find_residual_regions,
        suppress_coincident_marker_pairs,
    )

    edf      = detect_peaks(str(pattern_path), wavelength=wavelength,
                            instrument_profile=str(instrument_profile))
    obs_raw  = edf[["2theta", "intensity"]].values
    calc_raw = result.peak_data[["2theta", "intensity"]].values

    px    = np.asarray(result.plot_data.x)
    yobs  = np.asarray(result.plot_data.y_obs)
    ycalc = np.asarray(result.plot_data.y_calc)
    ybkg  = np.asarray(result.plot_data.y_bkg)

    pm = PeakMatcher(calc_raw, obs_raw, intensity_resolution=0.005,
                     profile_x=px, profile_y_calc=ycalc,
                     profile_y_obs=yobs, profile_y_bkg=ybkg)
    miss_f, extra_f = suppress_coincident_marker_pairs(
        pm.get_isolated_peaks("missing", min_intensity_ratio=missing_intensity_ratio),
        pm.get_isolated_peaks("extra",   min_intensity_ratio=extra_intensity_ratio),
    )

    parts = [a[:, 0] for a in (miss_f, extra_f) if len(a)]
    known = np.concatenate(parts) if parts else None

    residual = find_residual_regions(
        px, yobs, ycalc,
        profile_y_bkg=ybkg,
        matched_peak_positions=known,
        enabled=use_residual,
        window_detect_fraction=residual_window_detect_fraction,
        integral_fraction=residual_integral_fraction,
        calc_coverage_ratio=residual_calc_coverage_ratio,
    )

    miss_combined = np.vstack([miss_f, residual]) if len(residual) and len(miss_f) else (
        residual if len(residual) else miss_f
    )

    mismatch = find_intensity_mismatch_peaks(
        pm, px, yobs, ycalc, profile_y_bkg=ybkg,
        height_tolerance=intensity_mismatch_height_tolerance,
        area_tolerance=intensity_mismatch_area_tolerance,
    )

    result.refinement_metrics = RefinementMetrics(
        missing_peaks=miss_combined if len(miss_combined) else None,
        extra_peaks=extra_f if len(extra_f) else None,
        intensity_mismatch_peaks=mismatch if len(mismatch) else None,
        rwp=result.lst_data.rwp,
    )


def do_refinement(
    pattern_path: Path | str,
    phases: list[RefinementPhase | Path | str],
    wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float = "Cu",
    instrument_profile: str | Path = "Aeris-fds-Pixcel1d-Medipix3",
    working_dir: Path | str | None = None,
    phase_params: dict | None = None,
    refinement_params: dict | None = None,
    show_progress: bool = False,
    use_residual: bool = True,
    residual_integral_fraction: float = 0.010,
    residual_calc_coverage_ratio: float = 0.35,
    residual_window_detect_fraction: float = 0.003,
    missing_intensity_ratio: float = 0.005,
    extra_intensity_ratio: float = 0.03,
    intensity_mismatch_height_tolerance: float = 0.40,
    intensity_mismatch_area_tolerance: float = 0.60,
) -> RefinementResult:
    """Refine the structure using BGMN.

    Args:
        use_residual: see ``_attach_peak_markers``.
        residual_integral_fraction: see ``_attach_peak_markers``.
        residual_calc_coverage_ratio: see ``_attach_peak_markers``.
        residual_window_detect_fraction: see ``_attach_peak_markers``.
        missing_intensity_ratio: see ``_attach_peak_markers``.
        extra_intensity_ratio: see ``_attach_peak_markers``.
        intensity_mismatch_height_tolerance: see ``_attach_peak_markers``.
        intensity_mismatch_area_tolerance: see ``_attach_peak_markers``.
    """
    pattern_path = Path(pattern_path)
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

    if pattern_path.suffix.lower() not in (".xy",):
        pattern_path = convert_pattern_to_xy(pattern_path, working_dir)

    str_paths = []
    for phase_path in phases:
        phase = RefinementPhase.make(phase_path)
        phase_path_ = phase.path
        phase_params_ = phase_params.copy()
        phase_params_.update(phase.params)
        if phase_path_.suffix == ".cif":
            str_path = cif2str(phase_path_, "", working_dir, **phase_params_)
        else:
            if phase_path_.parent != working_dir:
                shutil.copy(phase_path_, working_dir)
            str_path = working_dir / phase_path_.name
        str_paths.append(str_path)

    control_file_path = generate_control_file(
        pattern_path=pattern_path,
        str_paths=str_paths,
        instrument_profile=instrument_profile,
        working_dir=working_dir,
        wavelength=wavelength,
        **refinement_params,
    )

    bgmn_worker = BGMNWorker()
    bgmn_worker.run_refinement_cmd(control_file_path, show_progress=show_progress)
    result = get_result(control_file_path)
    _attach_peak_markers(
        result, pattern_path, wavelength, instrument_profile,
        use_residual=use_residual,
        residual_integral_fraction=residual_integral_fraction,
        residual_calc_coverage_ratio=residual_calc_coverage_ratio,
        residual_window_detect_fraction=residual_window_detect_fraction,
        missing_intensity_ratio=missing_intensity_ratio,
        extra_intensity_ratio=extra_intensity_ratio,
        intensity_mismatch_height_tolerance=intensity_mismatch_height_tolerance,
        intensity_mismatch_area_tolerance=intensity_mismatch_area_tolerance,
    )
    return result


def do_refinement_no_saving(
    pattern_path: Path,
    phases: list[RefinementPhase | Path | str],
    wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float = "Cu",
    instrument_profile: str | Path = "Aeris-fds-Pixcel1d-Medipix3",
    phase_params: dict | None = None,
    refinement_params: dict | None = None,
    show_progress: bool = False,
    use_residual: bool = True,
    residual_integral_fraction: float = 0.010,
    residual_calc_coverage_ratio: float = 0.35,
    residual_window_detect_fraction: float = 0.003,
    missing_intensity_ratio: float = 0.005,
    extra_intensity_ratio: float = 0.03,
    intensity_mismatch_height_tolerance: float = 0.40,
    intensity_mismatch_area_tolerance: float = 0.60,
) -> RefinementResult:
    """Refine the structure using BGMN in a temporary directory without saving.

    Args:
        use_residual: see ``_attach_peak_markers``.
        residual_integral_fraction: see ``_attach_peak_markers``.
        residual_calc_coverage_ratio: see ``_attach_peak_markers``.
        residual_window_detect_fraction: see ``_attach_peak_markers``.
        missing_intensity_ratio: see ``_attach_peak_markers``.
        extra_intensity_ratio: see ``_attach_peak_markers``.
        intensity_mismatch_height_tolerance: see ``_attach_peak_markers``.
        intensity_mismatch_area_tolerance: see ``_attach_peak_markers``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        return do_refinement(
            pattern_path=pattern_path,
            phases=phases,
            wavelength=wavelength,
            instrument_profile=instrument_profile,
            working_dir=Path(tmpdir),
            phase_params=phase_params,
            refinement_params=refinement_params,
            show_progress=show_progress,
            use_residual=use_residual,
            residual_integral_fraction=residual_integral_fraction,
            residual_calc_coverage_ratio=residual_calc_coverage_ratio,
            residual_window_detect_fraction=residual_window_detect_fraction,
            missing_intensity_ratio=missing_intensity_ratio,
            extra_intensity_ratio=extra_intensity_ratio,
            intensity_mismatch_height_tolerance=intensity_mismatch_height_tolerance,
            intensity_mismatch_area_tolerance=intensity_mismatch_area_tolerance,
        )
