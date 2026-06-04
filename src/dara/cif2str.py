"""Convert CIF to Str format for BGMN."""
from __future__ import annotations

import datetime
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from asteval import Interpreter

from dara.utils import (
    POSSIBLE_SPECIES,
    fuzzy_compare,
    load_symmetrized_structure,
    process_phase_name,
    standardize_coords,
)

if TYPE_CHECKING:
    from pymatgen.core import Lattice
    from pymatgen.core.periodic_table import DummySpecie, Element, Specie
    from pymatgen.symmetry.structure import SymmetrizedStructure

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

LatticeSpec = Literal["fixed"] | float | int | tuple[float, float]
LatticeRange = LatticeSpec | dict[str, LatticeSpec]

class CIF2StrError(Exception):
    """CIF2Str error."""

def _normalize_lattice_spec(spec: Any, *, label: str) -> Literal["fixed"] | tuple[float, float]:
    """Normalize a single lattice spec into 'fixed' or (lo, hi) fractional deltas."""
    if spec == "fixed":
        return "fixed"
    if isinstance(spec, (int, float)):
        return (-float(spec), float(spec))
    try:
        lo, hi = spec
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid lattice spec for {label!r}: {spec!r}. "
            f"Expected 'fixed', a number, or a (lo, hi) tuple."
        )
    lo, hi = float(lo), float(hi)
    if lo > hi:
        raise ValueError(
            f"lattice_range lower bound ({lo}) must be <= upper bound ({hi}) for {label!r}"
        )
    return (lo, hi)

def process_specie_string(sp: str | Specie | Element | DummySpecie) -> str:
    """Reverse the charge notation of a species."""
    specie = re.sub(r"(\d+)([+-])", r"\2\1", str(sp))
    if specie.endswith(("+", "-")):
        specie += "1"
    specie = specie.upper()

    if specie not in POSSIBLE_SPECIES:
        # remove the valence and try again
        specie = re.search(r"[A-Z]+", specie).group(0)
        if specie not in POSSIBLE_SPECIES:
            raise CIF2StrError(
                f"Unknown species {specie}, the original specie string is {sp}"
            )
    return specie


def get_lattice_parameters_from_lattice(
    lattice: Lattice,
    crystal_system: Literal[
        "Monoclinic",
        "Cubic",
        "Hexagonal",
        "Trigonal",
        "Orthorhombic",
        "Triclinic",
        "Tetragonal",
        "Rhombohedral",
    ],
) -> dict[str, float]:
    """
    Get lattice parameters from lattice based on the type of lattice.

    .. note::
        The lattice parameters are in nm
    """
    if crystal_system == "Triclinic":
        return {
            "A": lattice.a / 10,
            "B": lattice.b / 10,
            "C": lattice.c / 10,
            "ALPHA": lattice.alpha,
            "BETA": lattice.beta,
            "GAMMA": lattice.gamma,
        }
    if crystal_system == "Monoclinic":
        return {
            "A": lattice.a / 10,
            "B": lattice.b / 10,
            "C": lattice.c / 10,
            "BETA": lattice.beta,
        }
    if crystal_system == "Orthorhombic":
        return {
            "A": lattice.a / 10,
            "B": lattice.b / 10,
            "C": lattice.c / 10,
        }
    if crystal_system == "Tetragonal":
        return {
            "A": lattice.a / 10,
            "C": lattice.c / 10,
        }
    if crystal_system == "Rhombohedral":
        return {
            "A": lattice.a / 10,
            "GAMMA": lattice.alpha,
        }
    # it seems that the trigonal and hexagonal lattices are the same in BGMN
    if crystal_system == "Hexagonal" or crystal_system == "Trigonal":
        return {
            "A": lattice.a / 10,
            "C": lattice.c / 10,
        }
    if crystal_system == "Cubic":
        return {
            "A": lattice.a / 10,
        }

    raise CIF2StrError(f"Unknown crystal system {crystal_system}")


def get_std_position(
    spacegroup_setting: dict[str, Any],
    wyckoff_letter: str,
    positions: list[list[float]],
) -> tuple[list[float], bool]:
    """Get the standard position of a site based on the hall number and wyckoff notation."""
    wyckoff = spacegroup_setting["wyckoffs"].get(wyckoff_letter, {})

    if not wyckoff:
        logger.debug(f"Spacegroup setting: {spacegroup_setting}")
        raise CIF2StrError(f"Cannot find the wyckoff letter {wyckoff_letter}")

    std_notations = wyckoff["std_notations"]

    positions = [standardize_coords(*position) for position in positions]

    for position in positions:
        variable_dict = {
            "x": position[0],
            "y": position[1],
            "z": position[2],
        }
        for std_notation in std_notations:
            constraints = std_notation.split(" ")

            aeval = Interpreter(use_numpy=False, symtable=variable_dict)
            wx, wy, wz = (aeval.eval(constraint) for constraint in constraints)
            logger.debug([position, (wx, wy, wz)])
            if (
                fuzzy_compare(wx, position[0])
                and fuzzy_compare(wy, position[1])
                and fuzzy_compare(wz, position[2])
            ):
                return position, True
    logger.debug(
        f"Cannot find the standard position for {wyckoff_letter} {std_notations}, using the first position. "
        f"The positions are: \n{positions}"
    )
    return positions[0], False


def check_wyckoff(
    spacegroup_setting: dict[str, Any], structure: SymmetrizedStructure
) -> tuple[list[dict[str, Any]], int]:
    """
    Check if a given spacegroup setting is valid for a structure.

    Args:
        spacegroup_setting: the spacegroup setting
        structure: the symmetrized structure

    Returns
    -------
        the settings of the elements and the number of errors
    """
    element_settings = []
    error_count = 0

    for site_idx in structure.equivalent_indices:
        idx = site_idx[0]
        site = structure[idx]
        wyckoff_letter = structure.wyckoff_letters[idx]
        if wyckoff_letter == "A":
            wyckoff_letter = "alpha"

        std_position, ok = get_std_position(
            spacegroup_setting,
            wyckoff_letter,
            [structure[idx].frac_coords for idx in site_idx],
        )

        if not ok:
            logger.debug(f"Site {site_idx} is not in the standard position")
            error_count += 1

        if site.is_ordered:
            species_string = process_specie_string(str(next(iter(site.species))))
        else:
            sorted_species = sorted(site.species)
            species_string = ",".join(
                f"{process_specie_string(ssp)}({site.species[ssp]:.6f})"
                for ssp in sorted_species
            )
            species_string = f"({species_string})"

        element_setting = {
            "E": species_string,
            "Wyckoff": wyckoff_letter,
            "x": f"{std_position[0]:.6f}",
            "y": f"{std_position[1]:.6f}",
            "z": f"{std_position[2]:.6f}",
            "TDS": f"{0.01:.6f}",
        }
        element_settings.append(element_setting)

    return element_settings, error_count


def make_spacegroup_setting_str(spacegroup_setting: dict[str, Any]) -> str:
    """Make the spacegroup setting string."""
    return (
        " ".join([f"{k}={v}" for k, v in spacegroup_setting["setting"].items()]) + " //"
    )


def make_lattice_parameters_str(
    spacegroup_setting: dict[str, Any],
    structure: SymmetrizedStructure,
    lattice_range: LatticeRange,
) -> str:
    """Make the lattice parameters string.

    Args:
        spacegroup_setting: the spacegroup setting dict.
        structure: the symmetrized structure.
        lattice_range: controls the refinement bounds for lattice parameters. Can be:
            - "fixed": all parameters are held fixed (not refined).
            - float `r`: symmetric fractional range for all params, i.e. bounds
              are `[v * (1 - r), v * (1 + r)]`.
            - tuple `(lo, hi)`: explicit fractional deltas for all params, bounds
              are `[v * (1 + lo), v * (1 + hi)]`.
            - dict mapping parameter name (e.g. "A", "B", "C", "ALPHA", "BETA",
              "GAMMA", case-insensitive) to any of the above per-parameter specs.
              Parameters not present in the dict fall back to the value under the
              "*" key, or to symmetric 0.1 if no "*" key is given.

        For tuple specs, if the unmodified value `v` falls outside the resulting
        window, the starting value is clamped to the boundary closest to `v`, so
        BGMN always receives a valid `lo <= start <= hi`.
    """
    crystal_system = spacegroup_setting["setting"]["Lattice"]
    lattice_parameters = get_lattice_parameters_from_lattice(
        structure.lattice, crystal_system
    )

    # build a per-parameter resolver
    if isinstance(lattice_range, dict):
        # normalize keys to uppercase so users can pass "a", "alpha", etc.
        norm_range = {}
        for key, spec in lattice_range.items():
            norm_key = key if key == "*" else key.upper()
            if norm_key in norm_range:
                raise ValueError(
                    f"Duplicate lattice parameter key after normalization: {norm_key!r}"
                )
            norm_range[norm_key] = spec

        default_spec = norm_range.get("*", 0.1)

        def resolve(name: str) -> Literal["fixed"] | tuple[float, float]:
            spec = norm_range.get(name, default_spec)
            return _normalize_lattice_spec(spec, label=name)
    else:
        normalized = _normalize_lattice_spec(lattice_range, label="lattice_range")

        def resolve(name: str) -> Literal["fixed"] | tuple[float, float]:
            return normalized

    parts = []
    for k, v in lattice_parameters.items():
        bounds = resolve(k)
        if bounds == "fixed":
            parts.append(f"{k}={v:.5f}")
        else:
            lo, hi = bounds
            if lo == hi:
                parts.append(f"{k}={v * (1 + lo):.5f}")
            else:
                lo_bound = v * (1 + lo)
                hi_bound = v * (1 + hi)
                # clamp the starting value into [lo_bound, hi_bound]
                start = min(max(v, lo_bound), hi_bound)
                parts.append(f"PARAM={k}={start:.5f}_{lo_bound:.5f}^{hi_bound:.5f}")

    return " ".join(parts) + " //"



_NUMERIC_GEWICHT_RE = re.compile(
    r"^-?\d+(?:\.\d+)?_-?\d+(?:\.\d+)?(?:\^-?\d+(?:\.\d+)?)?$"
)

def make_peak_parameter_str(k1: str, k2: str, b1: str, gewicht: str, rp: int) -> str:
    """Make the peak parameter string."""
    # Numeric specs are refinable; symbolic specs are emitted as fixed names.
    gewicht_part = (
        f"PARAM=GEWICHT={gewicht} //"
        if _NUMERIC_GEWICHT_RE.match(gewicht)
        else f"GEWICHT={gewicht} //"
    )

    return (
        f"RP={rp} "
        + (f"PARAM=k1={k1} " if k1 != "fixed" else "k1=0 ")
        + (f"PARAM=k2={k2} " if k2 != "fixed" else "k2=0 ")
        + (f"PARAM=B1={b1} " if b1 != "fixed" else "B1=0 ")
        + gewicht_part
    )


def cif2str(
    cif_path: Path,
    phase_name_suffix: str = "",
    working_dir: Path | None = None,
    *,
    lattice_range: LatticeRange = 0.1,
    gewicht: str = "0_0",
    rp: int = 4,
    k1: str = "0_0^0.01",
    k2: str = "0_0^0.01",
    b1: str = "0_0^0.01",
    lebail: bool = False,
    custom_params: list[str] | None = None,
    custom_params_map: dict[str, dict] | None = None,
) -> Path:
    """
    Convert CIF to Str format.

    Args:
        cif_path: the path to the CIF file
        phase_name_suffix: the suffix of the phase name
        working_dir: the folder to hold the processed str file
        lattice_range: controls the refinement bounds for the lattice parameters. Can be:
            - a single float `r` (default behavior): symmetric range `[a - r*a, a + r*a]`
            - the string "fixed": all lattice parameters are held fixed (not refined)
            - a tuple `(lo, hi)`: explicit fractional deltas, allowing asymmetric or
              one-sided ranges. For example:
                * `(-0.1, 0.1)` is equivalent to `0.1` (symmetric)
                * `(0.0, 0.1)` confines refinement to only positive deviations
                * `(-0.1, 0.0)` confines refinement to only negative deviations
                * `(-0.05, 0.2)` allows asymmetric refinement
            - a dict mapping a lattice parameter name to any of the above per-parameter
              specs, for fine-grained per-parameter control. Valid keys are "A", "B",
              "C", "ALPHA", "BETA", "GAMMA" (case-insensitive). Use the wildcard key "*"
              to set a fallback for any parameter not explicitly listed; if no "*" key is
              given, unlisted parameters default to a symmetric 0.1. For example:
                * `{"C": "fixed", "A": (-0.05, 0.2), "*": 0.1}` fixes C, refines A
                  asymmetrically, and refines the rest symmetrically at 0.1
                * `{"A": 0.1, "*": "fixed"}` refines only A and fixes everything else
        gewicht: the weight fraction of the phase to be refined. Options: 0_0, SPHAR0, and SPHAR2. If 0_0, then no
            preferred orientation. Read more in the BGMN manual.
        rp: the peak function to be used in the refinement. Read more in the BGMN manual.
        k1: the first peak parameter to be refined. Read more in the BGMN manual.
        k2: the second peak parameter to be refined. Read more in the BGMN manual.
        b1: the third peak parameter to be refined. Read more in the BGMN manual.
        lebail: whether to use the Le Bail method
        custom_params: optional list of custom BGMN string parameters to inject. This allows for defining complex
            mathematical equations, global parameters, or fractional occupancies
            (e.g., ["PARAM=Bglobal=0.05_0.01^0.20 //", "PARAM=BO=0.1_0.02^0.3 //"]).
        custom_params_map: optional dictionary mapping element symbols to dictionaries of parameters to add
            or overwrite. You can use the wildcard key "*" to apply parameters to all elements that are not
            specifically matched in the dictionary (e.g., {"*": {"TDS": "Bglobal"}, "O": {"TDS": "BO", "Occ": "OccO"}}).

    An example of the output .str file when using
    custom_params=["PARAM=Bglobal=0.05_0.01^0.20 //", "PARAM=BO=0.1_0.02^0.3 //"]
    and custom_params_map={"*": {"TDS": "Bglobal"}, "O": {"TDS": "BO", "Occ": "OccO"}}:

    PHASE=BariumzirconiumtinIVoxide105053 // ICSD_43137
    Reference=ICSD_43137 //
    Formula=Ba1_O3_Sn0.5_Zr0.5 //
    SpacegroupNo=221 HermannMauguin=P4/m-32/m Setting=1 Lattice=Cubic //
    PARAM=A=0.416280_0.412117^0.420443 //
    RP=4 k1=0 k2=0 PARAM=B1=0_0^0.01 GEWICHT=SPHAR4 //
    GOAL:BariumzirconiumtinIVoxide105053=GEWICHT*ifthenelse(ifdef(d),exp(my*d*3/4),1) //
    PARAM=Bglobal=0.05_0.01^0.20 //
    PARAM=BO=0.1_0.02^0.3 //
    E=BA+2 Wyckoff=b x=0.500000 y=0.500000 z=0.500000 TDS=Bglobal
    E=(ZR+4(0.5000),SN+4(0.5000)) Wyckoff=a x=0.000000 y=0.000000 z=0.000000 TDS=Bglobal
    E=O-2 Wyckoff=d x=0.500000 y=0.000000 z=0.000000 TDS=BO Occ=OccO

    """
    str_path = (
        cif_path.parent / f"{cif_path.stem}.str"
        if working_dir is None
        else working_dir / f"{cif_path.stem}.str"
    )

    structure, spg = load_symmetrized_structure(cif_path)

    hall_number = str(spg.get_symmetry_dataset().hall_number)
    with (Path(__file__).parent / "data" / "spglib_db" / "spg.json").open(
        "r", encoding="utf-8"
    ) as f:
        spg_group_db = json.load(f)
    settings = spg_group_db[hall_number]["settings"]

    best_setting = None
    for spacegroup_setting in settings:
        element_settings, error_count = check_wyckoff(spacegroup_setting, structure)
        if best_setting is None or error_count < best_setting[2]:
            best_setting = (spacegroup_setting, element_settings, error_count)

        if error_count == 0:
            break

    spacegroup_setting, element_settings, error_count = best_setting

    if error_count > 0:
        logger.debug(f"CIF file: {cif_path.read_text()}")
        logger.debug(f"Symmetry dataset: {spg.get_symmetry_dataset()}")
        raise CIF2StrError(
            f"Cannot find a valid lattice symmetry setting for {cif_path}."
        )

    logger.debug(
        f"Using setting {spacegroup_setting['setting']} for {cif_path}, with {error_count} errors"
    )

    # start to construct the str file string
    str_text = ""

    # add some metadata
    phase_name = process_phase_name(cif_path.stem + phase_name_suffix)
    str_text += f"PHASE={phase_name} // generated by pymatgen {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    formula = structure.composition.reduced_formula
    str_text += f"FORMULA={formula} //\n"

    # add spacegroup setting
    str_text += make_spacegroup_setting_str(spacegroup_setting) + "\n"

    # add lattice
    str_text += (
        make_lattice_parameters_str(
            spacegroup_setting, structure, lattice_range=lattice_range
        )
        + "\n"
    )

    # add RP
    str_text += make_peak_parameter_str(k1, k2, b1, gewicht, rp) + "\n"

    # add lebail
    if lebail:
        str_text += "LeBail=1\n"

    # add goals
    str_text += f"GOAL:{phase_name}=GEWICHT*ifthenelse(ifdef(d),exp(my*d*3/4),1) //\nGOAL=GrainSize(1,1,1) //\n"

    # add custom params injected from python dict
    if custom_params is not None:
        for line in custom_params:
            str_text += f"{line}\n"

    if custom_params_map is None:
        custom_params_map = {}

    # add wyckoff positions and overwrite parameters if mapped
    element_settings_str = []
    for element_setting in element_settings:
        element_name = element_setting.get("E", "")

        assigned_dict = None

        # First, check if there is a specific match for this element
        for element_key, custom_dict in custom_params_map.items():
            # Use regex with word boundaries to prevent partial matches
            # e.g., ensures "O" doesn't match the "O" in "CO" (Cobalt)
            if element_key != "*" and re.search(rf"\b{element_key}\b", element_name, re.IGNORECASE):
                assigned_dict = custom_dict
                break

        # If no specific match was found, check if a wildcard was provided
        if assigned_dict is None and "*" in custom_params_map:
            assigned_dict = custom_params_map["*"]

        # If we found either a specific match or a wildcard, update the dictionary
        if assigned_dict is not None:
            element_setting.update(assigned_dict)

        line_str = " ".join([f"{k}={v}" for k, v in element_setting.items()])
        element_settings_str.append(line_str)

    str_text += "\n".join(element_settings_str)

    with open(str_path, "w") as f:
        f.write(str_text)

    return str_path
