"""Utility functions for the dara package."""

from __future__ import annotations

import itertools
import logging
import os
import random
import re
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
from monty.json import MontyDecoder
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure
from scipy import signal

if TYPE_CHECKING:
    import pandas as pd
    from maggma.stores.mongolike import MongoStore


DEPRECATED = "DEPRECATED"

with open(Path(__file__).parent / "data" / "possible_species.txt") as f:
    POSSIBLE_SPECIES = {sp.strip() for sp in f}


def process_phase_name(phase_name: str, max_length: int | None = 10) -> str:
    """Process the phase name to remove special characters."""
    processed_name = re.sub(r"[\s()_/\\+–\-*.]", "", phase_name)
    if processed_name.isdigit():
        processed_name = f"P{processed_name}"
    # if empty, return UnknownPhase + some random alphabet
    if not processed_name:
        processed_name = "P" + "".join(
            random.choices(
                list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
                k=5,
            )
        )
    if max_length is not None:
        processed_name = processed_name[:max_length]
    return processed_name


def bool2yn(value: bool) -> str:
    """Convert boolean to Y (yes) or N (no)."""
    return "Y" if value else "N"


def get_number(s: Union[float, None, tuple[float, float]]) -> Union[float, None]:
    """Get the number from a float or tuple of floats."""
    if isinstance(s, tuple):
        return s[0]
    else:
        return s


def load_symmetrized_structure(
    cif_path: Path,
) -> tuple[SymmetrizedStructure, SpacegroupAnalyzer]:
    # suppress the warnings from pymatgen
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        structure = SpacegroupAnalyzer(
            Structure.from_file(
                cif_path.as_posix(), site_tolerance=1e-3, occupancy_tolerance=100
            )
        ).get_refined_structure()

    spg = SpacegroupAnalyzer(structure)
    structure: SymmetrizedStructure = spg.get_symmetrized_structure()
    return structure, spg


def get_optimal_max_two_theta(
    peak_data: pd.DataFrame, fraction: float = 0.7, intensity_filter=0.1
) -> float:
    """Get the optimal 2theta max given detected peaks. The range is determined by
    proportion of the detected peaks.

    Args:
        fraction: The fraction of the detected peaks. Defaults to 0.7.
        intensity_filter: The intensity filter; the fraction of the max intensity that
            is required for a peak to be acknowledged in this analysis. Defaults to 0.1.

    Returns
    -------
        A tuple of the optimal 2theta range.
    """
    max_intensity = peak_data.intensity.max()
    peak_data = peak_data[peak_data.intensity > intensity_filter * max_intensity]
    peak_data = peak_data.sort_values("2theta")
    peak_data = peak_data.reset_index(drop=True)

    num_peaks = len(peak_data)
    end_idx = round(fraction * num_peaks) - 1
    if end_idx < 0:
        end_idx = 0

    buffer = 1  # include the full last peak
    return round(peak_data["2theta"].iloc[end_idx], 2) + buffer


def read_phase_name_from_str(str_path: Path) -> str:
    """Get the phase name from the str file path.

    Example of str:
    PHASE=BaSnO3 // generated from pymatgen
    FORMULA=BaSnO3 //
    Lattice=Cubic HermannMauguin=P4/m-32/m Setting=1 SpacegroupNo=221 //
    PARAM=A=0.41168_0.40756^0.41580 //
    RP=4 PARAM=k1=0_0^1 k2=0 PARAM=B1=0_0^0.01 PARAM=GEWICHT=0_0 //
    GOAL:BaSnO3=GEWICHT //
    GOAL=GrainSize(1,1,1) //
    E=BA+2 Wyckoff=b x=0.500000 y=0.500000 z=0.500000 TDS=0.010000
    E=SN+4 Wyckoff=a x=0.000000 y=0.000000 z=0.000000 TDS=0.010000
    E=O-2 Wyckoff=d x=0.000000 y=0.000000 z=0.500000 TDS=0.010000
    """
    text = str_path.read_text()
    try:
        return re.search(r"PHASE=(\S*)", text).group(1)
    except AttributeError as e:
        raise ValueError(
            f"Could not find phase name in {str_path}. The content is: {text}"
        ) from e


def standardize_coords(x, y, z):
    # Adjust coordinates to specific fractional values if close
    fractions = {
        0.3333: 1 / 3,
        0.6667: 2 / 3,
        0.1667: 1 / 6,
        0.8333: 5 / 6,
        0.0833: 1 / 12,
        0.4167: 5 / 12,
        0.5833: 7 / 12,
        0.9167: 11 / 12,
    }

    for key, value in fractions.items():
        if abs(x - key) < 0.0001:
            x = value
        if abs(y - key) < 0.0001:
            y = value
        if abs(z - key) < 0.0001:
            z = value

    return x, y, z


def fuzzy_compare(a: float, b: float):
    fa = round(a, 6)
    fb = round(b, 6)

    # Normalizing the fractional parts to be within [0, 1]
    while fa < 0.0:
        fa += 1.0
    while fb < 0.0:
        fb += 1.0
    while fa >= 1.0:
        fa -= 1.0
    while fb >= 1.0:
        fb -= 1.0

    # Checking specific fractional values
    fractions = [
        (0.3333, 0.3334),  # 1/3
        (0.6666, 0.6667),  # 2/3
        (0.1666, 0.1667),  # 1/6
        (0.8333, 0.8334),  # 5/6
        (0.0833, 0.0834),  # 1/12
        (0.4166, 0.4167),  # 5/12
        (0.5833, 0.5834),  # 7/12
        (0.9166, 0.9167),  # 11/12
    ]

    for lower, upper in fractions:
        if lower <= fa <= upper and lower <= fb <= upper:
            return True

    # Fuzzy comparison for general case
    def is_close(_a, _b, rel_tol=0, abs_tol=1e-3):
        # Custom implementation of fuzzy comparison
        return abs(_a - _b) <= max(rel_tol * max(abs(_a), abs(_b)), abs_tol)

    return is_close(fa, fb)


def copy_and_rename_files(src_directory, dest_directory, file_map, verbose=True):
    """Copy specific files from the source directory to the destination directory with
    new names.

    :param src_directory: Path to the source directory
    :param dest_directory: Path to the destination directory
    :param file_map: Dictionary where keys are original filenames and values are new filenames
    """
    # Ensure the destination directory exists
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    # Copy and rename each specified file
    for src_filename, dest_filename in file_map.items():
        src_file = os.path.join(src_directory, src_filename)
        dest_file = os.path.join(dest_directory, dest_filename)

        # Check if file exists and is a file (not a directory)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dest_file)
            if verbose:
                print(
                    f"Successfully copied {src_filename} to {dest_filename} in {dest_directory}"
                )
        else:
            if verbose:
                print(f"ERROR: File {src_filename} not found in {src_directory}")


def get_chemsys_from_formulas(formulas: list[str]):
    """Convert a list of formulas to a chemsys."""
    elements = set()
    for formula in formulas:
        elements.update(Composition(formula).elements)

    return "-".join(sorted([str(e) for e in elements]))


def get_entries_in_chemsys_mp(chemsys: str):
    """Download ComputedStructureEntry objects from Materials Project."""
    try:
        from mp_api.client import MPRester
    except ImportError:
        raise ImportError("Please install the mp-api package.")

    with MPRester() as mpr:
        return mpr.get_entries_in_chemsys(chemsys)


def get_entries_db(db: MongoStore, chemsys: str):
    """Get entries for a specific chemical system from a database."""
    decoder = MontyDecoder()
    for doc in db.query(criteria={"chemsys": chemsys}, properties=["entry"]):
        yield decoder.process_decoded(doc["entry"])


def get_entries_in_chemsys_db(db: MongoStore, chemsys: list[str] | str):
    """Get all computed entries from a database covering all possible sub-chemical systems.

    This is equivalent to MPRester.get_entries_in_chemsys.

    Args:
        db: the database (must be connected!)
        chemsys: a chemical system, either as a string (e.g., "Li-Fe-O") or as a list of elements.

    """
    if isinstance(chemsys, str):
        elements = chemsys.split("-")
    else:
        elements = chemsys

    elements_set = set(elements)  # remove duplicate elements

    entries = []
    for i in range(len(elements_set)):
        for els in itertools.combinations(elements_set, i + 1):
            entries.extend(get_entries_db(db, "-".join(sorted(els))))

    return entries


def angular_correction(tt, eps1, eps2):
    deps1 = -eps1 * 360.0 / np.pi
    deps2 = 2.0 * (-eps2 * np.cos(tt * np.pi / 360)) * 180.0 / np.pi
    # deps3 = -eps3 * np.sin(np.deg2rad(tt)) * 180.0 / np.pi

    return deps1 + deps2  # + deps3


def intensity_correction(
    intensity: float,
    d_inv: float,
    gsum: float,
    wavelength: float,
    pol: float = 1,
    texture: float = 1,
):
    """
    Translated from Profex source (bgmnparparser.cpp:L112)
    Args:
        intensity: the intensity of the peak
        gsum: the gsum of the peak
        d_inv: the inverse of the d-spacing
        wavelength: the wavelength of the X-ray
        pol: the polarization factor, defaults to 1
        texture: the texture factor, defaults to 1

    Returns:
        the corrected intensity
    """
    # double sinx2 = std::pow(0.5 * dinv * pl.waveLength, 2.0);
    sinx2 = (0.5 * d_inv * wavelength) ** 2
    # theta = np.arcsin(np.sqrt(sinx2)) * 180.0 / np.pi
    # double intens = gsum * 360.0 * intens * 0.5 / (M_PI * std::sqrt(1.0 - sinx2) / pl.waveLength);
    # if (pl.polarization > 0.0) intens *= (0.5 * (1.0 + pl.polarization * std::pow(1.0 - 2.0 * sinx2, 2.0)));
    intensity = (
        gsum * 360.0 * intensity * 0.5 / (np.pi * np.sqrt(1.0 - sinx2) / wavelength)
    )
    if pol > 0.0:
        intensity *= 0.5 * (1.0 + pol * (1.0 - 2.0 * sinx2) ** 2.0)
    # intensity *= texture
    # intensity /= theta
    return intensity


def rwp(y_calc: np.ndarray, y_obs: np.ndarray) -> float:
    """
    Calculate the Rietveld weighted profile (RWP) for a refinement.

    The result is in percentage.

    Args:
        y_calc: the calculated intensity
        y_obs: the observed intensity
    Returns:
        the RWP
    """
    y_calc = np.array(y_calc)
    y_obs = np.array(y_obs)
    y_obs = np.clip(y_obs, 1e-6, None)
    return np.sqrt(np.sum((y_calc - y_obs) ** 2 / y_obs) / np.sum(y_obs)) * 100


def rpb(y_calc: np.ndarray, y_obs: np.ndarray, y_bkg) -> float:
    """
    Calculate the Rietveld profile without background (RPB) for a refinement.

    The result is in percentage.

    Args:
        y_calc: the calculated intensity
        y_obs: the observed intensity
    Returns:
        the RPB
    """
    y_calc = np.array(y_calc)
    y_obs = np.array(y_obs)
    return np.sum(np.abs(y_calc - y_obs)) / np.sum(np.abs(y_obs - y_bkg)) * 100


def get_logger(
    name: str,
    level=logging.DEBUG,
    log_format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
):
    """Code borrowed from the atomate package.

    Helper method for acquiring logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(log_format)

    if logger.hasHandlers():
        logger.handlers.clear()

    sh = logging.StreamHandler(stream=stream)
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.propagate = False

    return logger


def clean_icsd_code(icsd_code):
    """Add leading zeros to the ICSD code."""
    code = str(int(icsd_code))
    return (6 - len(code)) * "0" + code


def datetime_str() -> str:
    """Get a string representation of the current time."""
    return str(datetime.utcnow())


def find_optimal_score_threshold(
    scores: list[float] | np.ndarray,
) -> tuple[float, np.ndarray]:
    """Find the inflection point from a list of scores. We will calculate the percentile first"""
    if len(scores) == 0:
        return 0.0, np.array([]).reshape(-1)

    scores = np.array(scores)
    score_percentile = np.percentile(scores, np.arange(0, 101))
    score_percentile = signal.savgol_filter(score_percentile, 5, 1)

    second_derivative = np.diff(score_percentile, n=2)
    return score_percentile[np.argmax(second_derivative)].item(), score_percentile


def find_optimal_intensity_threshold(
    intensities: list[float] | np.ndarray, percentile: float = 90
) -> float:
    """
    Find the intensity threshold that captures percentile% of the intensities.

    Args:
        intensities: the list of intensities
        percentile: the percentile to capture, defaults to 90

    Returns:
        the intensity threshold
    """
    if len(intensities) == 0:
        return 0.0
    intensities_ = np.sort(intensities)[::-1]
    intensities_cum = np.cumsum(intensities_)
    intensities_cum /= intensities_cum[-1]
    idx = np.argmax(intensities_cum >= percentile / 100)
    return intensities_[idx]


def get_composition_from_filename(file_name: str | Path) -> Composition:
    """
    Get the composition from the filename. The composition is assumed to be the first
    part of the filename. For example, "BaSnO3_01.xrdml" will return "BaSnO3".
    """
    if isinstance(file_name, str):
        file_name = Path(file_name)

    return Composition(file_name.stem.split("_")[0])


def get_composition_distance(
    comp1: Composition | str, comp2: Composition | str, order: int = 2
) -> float:
    """
    Calculate the distance between two compositions.

    The default is the Manhattan.
    """
    comp1 = Composition(comp1, allow_negative=True).fractional_composition
    comp2 = Composition(comp2, allow_negative=True).fractional_composition

    delta_composition = comp1 - comp2
    delta_composition = {
        k: v / (comp1[k] + comp2[k]) for k, v in delta_composition.items()
    }

    return np.linalg.norm(np.array(list(delta_composition.values())), ord=order)
