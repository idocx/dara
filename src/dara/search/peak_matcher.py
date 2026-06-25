from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

DEFAULT_ANGLE_TOLERANCE = 0.2  # maximum difference in angle
DEFAULT_INTENSITY_TOLERANCE = 2  # maximum ratio of the intensities
# maximum ratio of the intensities to be considered as missing instead of wrong intensity
DEFAULT_MAX_INTENSITY_TOLERANCE = 5

WRONG_INTENSITY_HEIGHT_RATIO_THRESHOLD: float = 1 / 3
COINCIDENCE_SUPPRESSION_TOLERANCE: float = 0.06
DEFAULT_MISSING_MIN_INTENSITY_RATIO: float = 0.005
DEFAULT_EXTRA_MIN_INTENSITY_RATIO: float = 0.03
INTENSITY_MISMATCH_HEIGHT_TOLERANCE: float = 0.40  # height band [0.60, 1.40]; catches ~2x height mismatches
INTENSITY_MISMATCH_AREA_TOLERANCE: float = 0.60    # area band [0.40, 1.60]
INTENSITY_MISMATCH_WINDOW: float = 0.15             # degrees either side of peak apex for profile integration
RESIDUAL_WINDOW_WIDTH: float = 0.5    # degrees 2θ; sliding window width for integrated residual scan
RESIDUAL_WINDOW_STEP: float = 0.1    # degrees 2θ; step size between window centres
RESIDUAL_WINDOW_DETECT_FRACTION: float = 0.003  # per-window detection floor for forming/merging candidate regions
# merged-region threshold: flag only if total >= this fraction of total integrated obs
RESIDUAL_INTEGRAL_FRACTION: float = 0.010
# suppress merged region if integrated(calc-bkg)/integrated(obs-bkg) >= this
RESIDUAL_CALC_COVERAGE_RATIO: float = 0.35


def absolute_log_error(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the absolute error of two arrays in log space.

    Args:
        x: array 1
        y: array 2

    Returns
    -------
        the absolute error in log space
    """
    x = np.clip(x, 1e-10, None)
    y = np.clip(y, 1e-10, None)
    return np.abs(np.log(x) - np.log(y))


def distance_matrix(peaks1: np.ndarray, peaks2: np.ndarray) -> np.ndarray:
    """
    Return the distance matrix between two sets of peaks.

    The distance is defined as the maximum of the distance in position and the distance in intensity.
    The position distance is the absolute difference in position.
    The intensity distance is the absolute difference in log intensity.

    Args:
        peaks1: (n, 2) array of peaks with [position, intensity]
        peaks2: (m, 2) array of peaks with [position, intensity]

    Returns
    -------
        (n, m) distance matrix
    """
    position_distance = cdist(
        peaks1[:, 0].reshape(-1, 1), peaks2[:, 0].reshape(-1, 1), metric="cityblock"
    )
    intensity_distance = cdist(
        peaks1[:, 1].reshape(-1, 1),
        peaks2[:, 1].reshape(-1, 1),
        metric=absolute_log_error,
    )

    return np.sum(np.array([position_distance, intensity_distance]), axis=0)


def find_best_match(
    peak_calc: np.ndarray,
    peak_obs: np.ndarray,
    angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE,
    intensity_tolerance: float = DEFAULT_INTENSITY_TOLERANCE,
    max_intensity_tolerance: float = DEFAULT_MAX_INTENSITY_TOLERANCE,
) -> dict[str, Any]:
    """
    Find the best match between two sets of peaks.

    Args:
        peak_calc: the calculated peaks, (n, 2) array of peaks with [position, intensity]
        peak_obs: the observed peaks, (m, 2) array of peaks with [position, intensity]
        angle_tolerance: the maximum difference in angle
        intensity_tolerance: the maximum ratio of the intensities
        max_intensity_tolerance: the maximum ratio of the intensities to be considered as

    Returns
    -------
        missing[j]:
            the indices of the missing peaks in the ``obs peaks``
        matched[i, j]:
            the indices of both the matched peaks in the ``calculated peaks``
            and the ``observed peaks`` extra[i]: the indices of the extra peaks in the
            ``calculated peaks``
        wrong_intensity[i, j]:
            the indices of the peaks with wrong intensities in both
            the ``calculated peaks`` and the ``observed peaks``
        residual_peaks (N_peak_obs, 2):
            the residual peaks after matching (not including
            extra peaks in peak_calc)
    """
    matched = []
    extra = []
    wrong_intens = []

    if len(peak_obs) == 0:
        return {
            "missing": np.array([]).reshape(-1),
            "matched": np.array([]).reshape(-1, 2),
            "extra": np.arange(len(peak_calc)),
            "wrong_intensity": np.array([]).reshape(-1, 2),
            "residual_peaks": np.array([]).reshape(-1, 2),
        }
    if len(peak_calc) == 0:
        return {
            "missing": np.arange(len(peak_obs)),
            "matched": np.array([]).reshape(-1, 2),
            "extra": np.array([]).reshape(-1, 2),
            "wrong_intensity": np.array([]).reshape(-1, 2),
            "residual_peaks": peak_obs.copy(),
        }

    residual_peak_obs = peak_obs.copy()

    for peak_idx in np.argsort(peak_calc[:, 1])[::-1]:  # sort by intensity
        peak = peak_calc[peak_idx]

        all_close_obs_peaks_idx = np.where(
            np.abs(residual_peak_obs[:, 0] - peak[0]) <= angle_tolerance
        )[0]
        all_close_obs_peaks = residual_peak_obs[all_close_obs_peaks_idx]

        if len(all_close_obs_peaks) == 0:
            extra.append(peak_idx)
            continue

        best_match_idx = all_close_obs_peaks_idx[
            np.argmin(
                distance_matrix(np.array([peak]), all_close_obs_peaks).reshape(-1)
            )
        ]

        matched.append((peak_idx, best_match_idx))
        residual_peak_obs[best_match_idx, 1] -= peak[1]

    all_assigned = {m[1] for m in matched}
    missing = [i for i in range(len(peak_obs)) if i not in all_assigned]

    to_be_deleted = set()
    for i in range(len(matched)):
        peak_idx = matched[i][1]
        peak_intensity_diff = absolute_log_error(
            peak_obs[peak_idx][1],
            peak_obs[peak_idx][1] - residual_peak_obs[peak_idx][1],
        )
        if peak_intensity_diff > np.log(max_intensity_tolerance):
            missing.append(peak_idx)
            extra.append(matched[i][0])
            to_be_deleted.add(i)
        elif peak_intensity_diff > np.log(intensity_tolerance):
            wrong_intens.append(matched[i])
            to_be_deleted.add(i)

    matched = [m for i, m in enumerate(matched) if i not in to_be_deleted]

    return {
        "missing": missing,
        "matched": matched,
        "extra": extra,
        "wrong_intensity": wrong_intens,
        "residual_peaks": residual_peak_obs,
    }


def merge_peaks(peaks: np.ndarray, resolution: float = 0.0) -> np.ndarray:
    """
    Merge peaks that are too close to each other (smaller than resolution).

    Args:
        peaks: the peaks to merge
        resolution: the resolution to use for merging

    Returns
    -------
        the merged peaks
    """
    if len(peaks) <= 1 or resolution == 0.0:
        return peaks

    peaks = peaks[np.argsort(peaks[:, 0])]

    merge_to = np.arange(len(peaks))
    two_thetas = peaks[:, 0]

    for i in range(1, len(peaks)):
        two_theta_i = two_thetas[i]
        two_theta_im1 = two_thetas[i - 1]
        if np.abs(two_theta_im1 - two_theta_i) <= resolution:
            merge_to[i] = merge_to[i - 1]

    ptr_1 = ptr_2 = merge_to[0]
    new_peaks_list = []
    while ptr_1 < len(peaks):
        while ptr_2 < len(peaks) and merge_to[ptr_2] == ptr_1:
            ptr_2 += 1
        angles = peaks[ptr_1:ptr_2, 0]
        intensities = peaks[ptr_1:ptr_2, 1]

        updated_angle = angles @ intensities / np.sum(intensities)
        updated_intensity = np.sum(intensities)

        new_peaks_list.append([updated_angle, updated_intensity])

        ptr_1 = ptr_2

    return np.array(new_peaks_list)


class PeakMatcher:
    """
    Peak matcher class to match the calculated peaks with the observed peaks.

    Args:
        peak_calc: the calculated peaks, (n, 2) array of peaks with [position, intensity]
        peak_obs: the observed peaks, (m, 2) array of peaks with [position, intensity]
        intensity_resolution: the resolution for the intensity, default to 0.01. Filter out peaks with lower intensity
        angle_resolution: the resolution for the angle, default to 0.1
        angle_tolerance: the maximum difference in angle, default to 0.3
        intensity_tolerance: the maximum ratio of the intensities, default to 2
        max_intensity_tolerance: the maximum ratio of the intensities to be considered as missing or extra,
            default to 10
        profile_x: 2θ grid (degrees) for the full diffraction profile. When this and
            the three profile_y_* args are all given, matched peaks are re-examined
            against the profile (see ``_reclassify_wrong_intensity_by_height`` and
            ``_relocate_extra_to_apex``); omit all four to skip this step.
        profile_y_calc: calculated profile on ``profile_x``.
        profile_y_obs: observed profile on ``profile_x``.
        profile_y_bkg: background profile on ``profile_x``.
        height_ratio_threshold: passed to ``_reclassify_wrong_intensity_by_height``;
            wrong_intensity pairs whose calc/obs profile-height ratio falls below
            this are reclassified as missing+extra instead.
    """

    def __init__(
        self,
        peak_calc: np.ndarray,
        peak_obs: np.ndarray,
        intensity_resolution: float = 0.01,
        angle_resolution: float = 0.1,
        angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE,
        intensity_tolerance: float = DEFAULT_INTENSITY_TOLERANCE,
        max_intensity_tolerance: float = DEFAULT_MAX_INTENSITY_TOLERANCE,
        profile_x: np.ndarray | None = None,
        profile_y_calc: np.ndarray | None = None,
        profile_y_obs: np.ndarray | None = None,
        profile_y_bkg: np.ndarray | None = None,
        height_ratio_threshold: float = WRONG_INTENSITY_HEIGHT_RATIO_THRESHOLD,
    ):
        self.intensity_resolution = intensity_resolution
        self.angle_resolution = angle_resolution

        peak_calc = peak_calc.reshape(-1, 2)
        peak_obs = peak_obs.reshape(-1, 2)

        peak_calc = peak_calc[
            (peak_calc[:, 1] > 0)
            & (peak_calc[:, 1] > intensity_resolution * peak_calc[:, 1].max(initial=0))
        ]

        self.peak_calc = merge_peaks(peak_calc, resolution=angle_resolution)

        peak_obs = peak_obs[
            (peak_obs[:, 1] > 0)
            & (peak_obs[:, 1] > intensity_resolution * peak_obs[:, 1].max(initial=0))
        ]

        self.peak_obs = merge_peaks(peak_obs, resolution=angle_resolution)

        self._result = find_best_match(
            self.peak_calc,
            self.peak_obs,
            angle_tolerance=angle_tolerance,
            intensity_tolerance=intensity_tolerance,
            max_intensity_tolerance=max_intensity_tolerance,
        )

        if all(v is not None for v in
               [profile_x, profile_y_calc, profile_y_obs, profile_y_bkg]):
            px      = np.asarray(profile_x)
            py_calc = np.asarray(profile_y_calc)
            py_obs  = np.asarray(profile_y_obs)
            py_bkg  = np.asarray(profile_y_bkg)
            self._reclassify_wrong_intensity_by_height(
                px, py_calc, py_obs, py_bkg, height_ratio_threshold,
            )
            self._relocate_extra_to_apex(px, py_calc, py_bkg)

    def _reclassify_wrong_intensity_by_height(
        self,
        profile_x: np.ndarray,
        profile_y_calc: np.ndarray,
        profile_y_obs: np.ndarray,
        profile_y_bkg: np.ndarray,
        height_ratio_threshold: float,
    ) -> None:
        """Re-examine wrong_intensity pairs using background-subtracted profile heights.

        When the ratio calc_height / obs_height is below *height_ratio_threshold*
        the calc peak is too weak relative to the observed feature to count as
        an explanation.  The pair is reclassified: obs peak -> missing,
        calc peak -> extra.  Pairs that pass the height check remain wrong_intensity.
        """
        keep_wi: list = []
        for c_idx, o_idx in self._result["wrong_intensity"]:
            tt_c = float(self.peak_calc[c_idx, 0])
            tt_o = float(self.peak_obs[o_idx, 0])
            calc_h = float(np.interp(tt_c, profile_x, profile_y_calc - profile_y_bkg))
            obs_h  = float(np.interp(tt_o, profile_x, profile_y_obs  - profile_y_bkg))
            if obs_h > 0 and calc_h / obs_h < height_ratio_threshold:
                if o_idx not in self._result["missing"]:
                    self._result["missing"].append(o_idx)
                if c_idx not in self._result["extra"]:
                    self._result["extra"].append(c_idx)
            else:
                keep_wi.append((c_idx, o_idx))
        self._result["wrong_intensity"] = keep_wi

    def _relocate_extra_to_apex(
        self,
        profile_x: np.ndarray,
        profile_y_calc: np.ndarray,
        profile_y_bkg: np.ndarray,
        apex_window: float = 0.3,
    ) -> None:
        """Replace centroid 2θ of extra-calc peaks with the nearest profile-apex 2θ.

        merge_peaks() uses an intensity-weighted centroid for merged sub-peaks, which
        can land between two profile maxima and corrupt coincidence-suppression distances.
        Each extra peak is reassigned to the nearest local maximum of the
        background-subtracted calc profile within *apex_window* degrees.
        Peaks with no local maximum in their window keep their centroid position.
        """
        net = profile_y_calc - profile_y_bkg
        dx = float(profile_x[1] - profile_x[0]) if len(profile_x) > 1 else 0.01
        min_dist_pts = max(1, int(0.05 / dx))
        local_max_pts, _ = find_peaks(net, distance=min_dist_pts)
        if len(local_max_pts) == 0:
            return
        local_max_x = profile_x[local_max_pts]
        for c_idx in self._result["extra"]:
            tt_old = float(self.peak_calc[c_idx, 0])
            in_window = (local_max_x >= tt_old - apex_window) & (local_max_x <= tt_old + apex_window)
            candidates = local_max_x[in_window]
            if len(candidates) == 0:
                continue
            self.peak_calc[c_idx, 0] = float(candidates[np.argmin(np.abs(candidates - tt_old))])

    @property
    def missing(self) -> np.ndarray:
        """Get the missing peaks in the `observed peaks`. The shape should be (N, 2) with [position, intensity]."""
        missing = self._result["missing"]
        missing = np.array(missing).reshape(-1)
        return (
            self.peak_obs[missing] if len(missing) > 0 else np.array([]).reshape(-1, 2)
        )

    @property
    def matched(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the matched peaks in both the `calculated peaks` and the `observed peaks`."""
        matched = self._result["matched"]
        matched = np.array(matched).reshape(-1, 2)
        return (
            self.peak_calc[matched[:, 0]]
            if len(matched) > 0
            else np.array([]).reshape(-1, 2),
            self.peak_obs[matched[:, 1]]
            if len(matched) > 0
            else np.array([]).reshape(-1, 2),
        )

    @property
    def extra(self) -> np.ndarray:
        """Get the extra peaks in the `calculated peaks`."""
        extra = self._result["extra"]
        extra = np.array(extra).reshape(-1)
        return self.peak_calc[extra] if len(extra) > 0 else np.array([]).reshape(-1, 2)

    @property
    def wrong_intensity(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the indices of the peaks with wrong intensities in both the
        `calculated peaks` and the `observed peaks`.
        """
        wrong_intens = self._result["wrong_intensity"]
        wrong_intens = np.array(wrong_intens).reshape(-1, 2)

        return (
            self.peak_calc[np.array(wrong_intens)[:, 0]]
            if len(wrong_intens) > 0
            else np.array([]).reshape(-1, 2),
            self.peak_obs[np.array(wrong_intens)[:, 1]]
            if len(wrong_intens) > 0
            else np.array([]).reshape(-1, 2),
        )

    def score(
        self,
        matched_coeff: float = 1,
        wrong_intensity_coeff: float = 1,
        missing_coeff: float = -0.01,
        extra_coeff: float = -1,
        normalize: bool = True,
    ) -> float:
        """
        Calculate the score of the matching result.

        Args:
            matched_coeff: the coefficient of the matched peaks
            wrong_intensity_coeff: the coefficient of the peaks with wrong intensities
            missing_coeff: the coefficient of the missing peaks
            extra_coeff: the coefficient of the extra peaks
            normalize: whether to normalize the score by the total intensity of the observed peaks

        Returns
        -------
            the score of the matching result
        """
        matched_obs, matched_calc = self.matched
        wrong_intens_obs, wrong_intens_calc = self.wrong_intensity
        matched_peaks = min([matched_obs, matched_calc], key=lambda x: x[:, 1].sum())
        wrong_intens_peaks = min(
            [wrong_intens_obs, wrong_intens_calc], key=lambda x: x[:, 1].sum()
        )
        missing_obs = self.missing
        extra_calc = self.extra

        score = (
            np.sum(np.abs(matched_peaks[:, 1])) * matched_coeff
            + np.sum(np.abs(wrong_intens_peaks[:, 1])) * wrong_intensity_coeff
            + np.sum(np.abs(extra_calc[:, 1])) * extra_coeff
            + np.sum(np.abs(missing_obs[:, 1])) * missing_coeff
        )

        if normalize:
            total_peak_obs = np.sum(np.abs(self.peak_obs[:, 1]))
            score /= total_peak_obs

        return score

    def jaccard_index(self) -> float:
        """
        Calculate the Jaccard index of the matching result.

        Returns
        -------
            the Jaccard index of the matching result
        """
        matched_calc = self.matched[0]
        wrong_intens_calc = self.wrong_intensity[0]
        matched_obs = self.matched[1]
        wrong_intens_obs = self.wrong_intensity[1]

        total_intensity = np.sum(np.abs(self.peak_obs[:, 1])) + np.sum(
            np.abs(self.peak_calc[:, 1])
        )

        matched_intensity = np.sum(np.abs(matched_calc[:, 1])) + np.sum(
            np.abs(matched_obs[:, 1])
        )
        wrong_intens_intensity = np.sum(np.abs(wrong_intens_calc[:, 1])) + np.sum(
            np.abs(wrong_intens_obs[:, 1])
        )

        if total_intensity == 0:
            return 0

        return (matched_intensity + wrong_intens_intensity) / total_intensity

    def get_isolated_peaks(
        self,
        peak_type: Literal["missing", "extra"],
        min_angle_difference: float = 0.3,
        min_intensity_ratio: float | None = None,
    ) -> np.ndarray:
        """
        Get the isolated missing peaks in the `observed peaks`.

        The isolated missing/extra peaks are the missing/extra peaks that are not close to any other
        peaks in matched and wrong intensity peaks.

        Args:
            peak_type: the type of the peaks to consider, either "missing" or "extra"
            min_angle_difference: the tolerance to consider a peak as close to another peak, default to 0.3 degree
            min_intensity_ratio: minimum intensity relative to the observed maximum.
                Defaults to DEFAULT_MISSING_MIN_INTENSITY_RATIO for missing peaks and
                DEFAULT_EXTRA_MIN_INTENSITY_RATIO for extra peaks when None.

        Returns
        -------
            the isolated missing peaks with [position, intensity]
        """
        if min_intensity_ratio is None:
            min_intensity_ratio = (
                DEFAULT_MISSING_MIN_INTENSITY_RATIO
                if peak_type == "missing"
                else DEFAULT_EXTRA_MIN_INTENSITY_RATIO
            )

        if peak_type == "missing":
            peaks = self.missing
            matched = self.matched[1]
            wrong_intens = self.wrong_intensity[1]
        else:
            peaks = self.extra
            matched = self.matched[0]
            wrong_intens = self.wrong_intensity[0]

        matched = np.concatenate([matched, wrong_intens])

        if len(peaks) == 0:
            return np.array([]).reshape(-1, 2)
        if len(matched) == 0:
            return peaks[peaks[:, 1] > min_intensity_ratio * self.peak_obs[:, 1].max()]

        distance = cdist(
            peaks[:, 0].reshape(-1, 1),
            matched[:, 0].reshape(-1, 1),
            metric="cityblock",
        )
        distance = np.min(distance, axis=1)
        min_intensity = self.peak_obs[:, 1].max() * min_intensity_ratio

        return peaks[(distance > min_angle_difference) & (peaks[:, 1] > min_intensity)]

    def visualize(self):
        import matplotlib.pyplot as plt

        missing_obs = self.missing
        matched_obs = self.matched[1]
        wrong_intensity_obs = self.wrong_intensity[1]

        extra_calc = self.extra
        matched_calc = self.matched[0]
        wrong_intensity_calc = self.wrong_intensity[0]

        extra_calc[:, 1] *= -1
        matched_calc[:, 1] *= -1
        wrong_intensity_calc[:, 1] *= -1

        extra_peaks = extra_calc
        missing_peaks = missing_obs
        matched_peaks = np.concatenate([matched_calc, matched_obs])
        wrong_intensity_peaks = np.concatenate(
            [wrong_intensity_calc, wrong_intensity_obs]
        )

        _, ax = plt.subplots()

        ax.vlines(
            missing_peaks[:, 0],
            0,
            missing_peaks[:, 1],
            color="red",
            alpha=0.5,
            label="missing",
        )
        ax.vlines(
            matched_peaks[:, 0],
            0,
            matched_peaks[:, 1],
            color="green",
            alpha=0.5,
            label="matched",
        )
        ax.vlines(
            extra_peaks[:, 0],
            0,
            extra_peaks[:, 1],
            color="blue",
            alpha=0.5,
            label="extra",
        )
        ax.vlines(
            wrong_intensity_peaks[:, 0],
            0,
            wrong_intensity_peaks[:, 1],
            color="orange",
            alpha=0.5,
            label="wrong intens",
        )

        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("2theta")
        ax.set_ylabel("Intensity")
        ax.legend()


def suppress_coincident_marker_pairs(
    missing_peaks: np.ndarray,
    extra_peaks: np.ndarray,
    tolerance: float = COINCIDENCE_SUPPRESSION_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Suppress missing/extra marker pairs within *tolerance* degrees of each other.

    Near-coincident missing+extra pairs indicate refinement position wobble: the
    model placed a reflection slightly off-angle rather than genuinely missing it.
    Both markers are dropped.

    Returns
    -------
        (missing_peaks, extra_peaks) with coincident pairs removed, same shapes
    """
    missing_peaks = np.asarray(missing_peaks).reshape(-1, 2)
    extra_peaks   = np.asarray(extra_peaks).reshape(-1, 2)

    if len(missing_peaks) == 0 or len(extra_peaks) == 0:
        return missing_peaks, extra_peaks

    miss_suppress: set[int] = set()
    extra_suppress: set[int] = set()
    for m_i, (mt, _) in enumerate(missing_peaks):
        for e_i, (et, _) in enumerate(extra_peaks):
            if abs(float(mt) - float(et)) <= tolerance:
                miss_suppress.add(m_i)
                extra_suppress.add(e_i)

    keep_m = [i for i in range(len(missing_peaks)) if i not in miss_suppress]
    keep_e = [i for i in range(len(extra_peaks))   if i not in extra_suppress]

    out_m = missing_peaks[keep_m] if keep_m else np.empty((0, 2))
    out_e = extra_peaks[keep_e]   if keep_e else np.empty((0, 2))
    return out_m, out_e


def find_residual_regions(
    profile_x: np.ndarray,
    profile_y_obs: np.ndarray,
    profile_y_calc: np.ndarray,
    profile_y_bkg: np.ndarray | None = None,
    window_width: float = RESIDUAL_WINDOW_WIDTH,
    window_step: float = RESIDUAL_WINDOW_STEP,
    window_detect_fraction: float = RESIDUAL_WINDOW_DETECT_FRACTION,
    integral_fraction: float = RESIDUAL_INTEGRAL_FRACTION,
    calc_profile_ratio: float = DEFAULT_EXTRA_MIN_INTENSITY_RATIO,
    calc_coverage_ratio: float = RESIDUAL_CALC_COVERAGE_RATIO,
    matched_peak_positions: np.ndarray | None = None,
    enabled: bool = True,
) -> np.ndarray:
    """Find regions where the integrated positive residual indicates an unfit feature.

    Scans the pattern with a fixed-width sliding window.  Each window's integrated
    positive residual (sum of max(y_obs - y_calc, 0) * dx) is compared against a
    fraction of the total integrated observed intensity.  Overlapping flagged windows
    are merged into contiguous regions.

    Bragg-peak filter (corrected form): a merged region is suppressed only if the
    calc *profile* is already adequate inside it — i.e. max(y_calc - y_bkg) within
    the region exceeds ``calc_profile_ratio * max(y_obs)``.  A tabulated Bragg peak
    whose calc profile stays near background does NOT suppress the region.

    Args:
        profile_x: 2θ grid (degrees), uniformly spaced.
        profile_y_obs: observed counts on that grid.
        profile_y_calc: calculated profile on that grid.
        profile_y_bkg: background on that grid.  When None, zeros are used (filter
            compares calc profile directly against obs max).
        window_width: sliding window width in degrees (default 0.5°).
        window_step: step between window centres in degrees (default 0.1°).
        window_detect_fraction: per-window detection floor for forming candidate
            windows and merging them into regions (default 0.003 = 0.3%).  This
            is intentionally lower than ``integral_fraction`` so that broad
            features are not penalised for being diffuse.
        integral_fraction: merged-region threshold — a region is flagged only if
            its total integrated positive residual exceeds this fraction of the
            total integrated observed intensity (default 0.010 = 1.0%).
        calc_profile_ratio: suppress a window if max(y_calc - y_bkg) inside it
            exceeds this fraction of max(y_obs) (default 0.03 = 3%).
        calc_coverage_ratio: suppress a merged region when
            integrated(y_calc - y_bkg) / integrated(y_obs - y_bkg) over the
            merged span exceeds this fraction — the calc already accounts for
            this much of the observed area in that region (default 0.35).
        matched_peak_positions: 1-D array of 2θ positions of already-identified
            missing or extra peaks (after all normal suppression).  Any merged
            region whose [start, end] span contains one of these positions is
            dropped — the feature has already been flagged by the matcher.
        enabled: when False, skip all computation and return an empty (0, 4)
            array.  Default True reproduces normal behavior.

    Returns
    -------
        Array of shape (N, 2) with [center_2theta, 0.0] for each flagged region,
        formatted identically to missing-peak arrays for direct concatenation.
    """
    if not enabled:
        return np.empty((0, 2))

    profile_x      = np.asarray(profile_x, dtype=float)
    profile_y_obs  = np.asarray(profile_y_obs, dtype=float)
    profile_y_calc = np.asarray(profile_y_calc, dtype=float)
    if profile_y_bkg is not None:
        profile_y_bkg = np.asarray(profile_y_bkg, dtype=float)
    else:
        profile_y_bkg = np.zeros_like(profile_y_obs)

    dx = float(profile_x[1] - profile_x[0]) if len(profile_x) > 1 else 0.01
    obs_max = float(profile_y_obs.max())
    half_w  = window_width / 2.0

    total_obs_integral = float(np.sum(np.maximum(profile_y_obs - profile_y_bkg, 0.0))) * dx
    detect_threshold = window_detect_fraction * total_obs_integral  # per-window floor
    region_threshold = integral_fraction * total_obs_integral        # merged-region gate

    positive_residual = np.maximum(profile_y_obs - profile_y_calc, 0.0)

    # Integrated-residual scan: flag broad features the matcher never detected.
    calc_profile_threshold = calc_profile_ratio * obs_max
    calc_net = profile_y_calc - profile_y_bkg
    obs_net  = np.maximum(profile_y_obs - profile_y_bkg, 0.0)

    flagged_centres: list[float] = []
    centre = float(profile_x[0]) + half_w
    while centre <= float(profile_x[-1]) - half_w:
        mask = (profile_x >= centre - half_w) & (profile_x <= centre + half_w)
        win_integral = float(np.sum(positive_residual[mask])) * dx
        calc_net_max = float(calc_net[mask].max())
        if win_integral >= detect_threshold and calc_net_max < calc_profile_threshold:
            flagged_centres.append(centre)
        centre += window_step

    if not flagged_centres:
        return np.empty((0, 2))

    merged: list[tuple[float, float]] = []
    span_start = flagged_centres[0] - half_w
    span_end   = flagged_centres[0] + half_w
    for c in flagged_centres[1:]:
        win_start = c - half_w
        win_end   = c + half_w
        if win_start <= span_end + window_step / 2:
            span_end = max(span_end, win_end)
        else:
            merged.append((span_start, span_end))
            span_start = win_start
            span_end   = win_end
    merged.append((span_start, span_end))

    known_positions: np.ndarray | None = None
    if matched_peak_positions is not None:
        known_positions = np.asarray(matched_peak_positions, dtype=float).reshape(-1)

    regions: list[list[float]] = []
    for s, e in merged:
        mask = (profile_x >= s) & (profile_x <= e)
        seg_res      = positive_residual[mask]
        merged_integ = float(np.sum(seg_res)) * dx
        if merged_integ < region_threshold:
            continue
        region_obs_int  = float(np.sum(obs_net[mask])) * dx
        region_calc_int = float(np.sum(np.maximum(calc_net[mask], 0.0))) * dx
        if region_obs_int > 0 and region_calc_int / region_obs_int >= calc_coverage_ratio:
            continue
        if known_positions is not None and np.any((known_positions >= s) & (known_positions <= e)):
            continue
        midpoint = (float(s) + float(e)) / 2.0
        regions.append([midpoint, 0.0])

    return np.array(regions).reshape(-1, 2) if regions else np.empty((0, 2))


def find_intensity_mismatch_peaks(
    pm: PeakMatcher,
    profile_x: np.ndarray,
    profile_y_obs: np.ndarray,
    profile_y_calc: np.ndarray,
    profile_y_bkg: np.ndarray | None = None,
    height_tolerance: float = INTENSITY_MISMATCH_HEIGHT_TOLERANCE,
    area_tolerance: float = INTENSITY_MISMATCH_AREA_TOLERANCE,
    window: float = INTENSITY_MISMATCH_WINDOW,
) -> np.ndarray:
    """Flag matched peaks whose profile height or area deviates beyond the respective band.

    Operates only on matched pairs — missing and extra take precedence.
    height band: [1 - height_tolerance, 1 + height_tolerance]
    area band:   [1 - area_tolerance,   1 + area_tolerance]
    Flags if either band is violated.

    Args:
        pm: the PeakMatcher holding the matched calc/obs peak pairs.
        profile_x: 2θ grid (degrees), uniformly spaced.
        profile_y_obs: observed counts on that grid.
        profile_y_calc: calculated profile on that grid.
        profile_y_bkg: background on that grid. When None, zeros are used.
        height_tolerance: half-width of the height band (default 0.40).
        area_tolerance: half-width of the area band (default 0.60).
        window: degrees either side of the peak apex used for height/area
            integration (default 0.15).

    Returns
    -------
        (N, 2) array of [obs_2theta, 0.0], same shape as missing_peaks
    """
    matched_pairs = pm._result["matched"]
    if not matched_pairs:
        return np.empty((0, 2))

    profile_x      = np.asarray(profile_x,      dtype=float)
    profile_y_obs  = np.asarray(profile_y_obs,  dtype=float)
    profile_y_calc = np.asarray(profile_y_calc, dtype=float)
    profile_y_bkg = np.zeros_like(profile_y_obs) if profile_y_bkg is None else np.asarray(profile_y_bkg, dtype=float)

    dx   = float(profile_x[1] - profile_x[0]) if len(profile_x) > 1 else 0.01
    h_lo = 1.0 - height_tolerance
    h_hi = 1.0 + height_tolerance
    a_lo = 1.0 - area_tolerance
    a_hi = 1.0 + area_tolerance

    seen_obs: set[int] = set()
    flagged: list[list[float]] = []

    for _c_idx, o_idx in matched_pairs:
        if o_idx in seen_obs:
            continue
        seen_obs.add(o_idx)

        tt_o = float(pm.peak_obs[o_idx, 0])

        mask     = (profile_x >= tt_o - window) & (profile_x <= tt_o + window)
        obs_net  = np.maximum(profile_y_obs[mask]  - profile_y_bkg[mask], 0.0)
        calc_net = np.maximum(profile_y_calc[mask] - profile_y_bkg[mask], 0.0)
        obs_area  = float(np.sum(obs_net))  * dx
        calc_area = float(np.sum(calc_net)) * dx

        if obs_area <= 0:
            continue

        obs_h  = float(np.interp(tt_o, profile_x, profile_y_obs  - profile_y_bkg))
        calc_h = float(np.interp(tt_o, profile_x, profile_y_calc - profile_y_bkg))

        if obs_h <= 0:
            continue

        area_ratio   = calc_area / obs_area
        height_ratio = calc_h   / obs_h

        if not (h_lo <= height_ratio <= h_hi) or not (a_lo <= area_ratio <= a_hi):
            flagged.append([tt_o, 0.0])

    return np.array(flagged).reshape(-1, 2) if flagged else np.empty((0, 2))
