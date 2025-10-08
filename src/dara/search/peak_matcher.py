from typing import Any, Literal

import numpy as np
from scipy.spatial.distance import cdist

DEFAULT_ANGLE_TOLERANCE = 0.2  # maximum difference in angle
DEFAULT_INTENSITY_TOLERANCE = 2  # maximum ratio of the intensities
# maximum ratio of the intensities to be considered as missing instead of wrong intensity
DEFAULT_MAX_INTENSITY_TOLERANCE = 5


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

    # tell if a peak has wrong intensity by the sum of the intensities of the matched peaks
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

    # sorted by 0th column
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
        missing_coeff: float = -0.1,
        extra_coeff: float = -0.5,
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
        min_intensity_ratio: float = 0.03,
    ) -> np.ndarray:
        """
        Get the isolated missing peaks in the `observed peaks`.

        The isolated missing/extra peaks are the missing/extra peaks that are not close to any other
        peaks in matched and wrong intensity peaks.

        Args:
            peak_type: the type of the peaks to consider, either "missing" or "extra"
            min_angle_difference: the tolerance to consider a peak as close to another peak, default to 0.3 degree
            min_intensity_ratio: the minimum ratio of the intensity to be considered as a peak, default to 0.01

        Returns
        -------
            the isolated missing peaks with [position, intensity]
        """
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

        # add a line y=0
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("2theta")
        ax.set_ylabel("Intensity")
        ax.legend()
