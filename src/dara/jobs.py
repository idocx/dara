"""Jobflow jobs for DARA."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from monty.serialization import dumpfn
from pymatgen.core import Composition

from dara.cif import Cif
from dara.icsd import ICSDDatabase
from dara.prediction.core import PhasePredictor
from dara.refine import do_refinement, do_refinement_no_saving
from dara.schema import PhaseSearchDocument, RefinementDocument
from dara.search import search_phases
from dara.search.data_model import SearchResult
from dara.utils import get_logger

if TYPE_CHECKING:
    from dara.xrd import XRDData

try:
    from jobflow import Maker, job
except ImportError:
    raise ImportError("Please install jobflow to use the dara jobs module!")


logger = get_logger(__name__)


@dataclass
class RefinementMaker(Maker):
    """Maker to create a job for performing Rietveld refinement using BGMN.

    Args:
        name: The name of the job. This is automatically created if not provided.
    """

    name: str = "refine"
    save: bool = True
    refinement_params: dict | None = None
    show_progress: bool = True

    @job(output_schema=RefinementDocument)
    def make(
        self,
        xrd_data: XRDData,
        cifs: list[Cif],
        phase_params: dict | None = None,
        instrument_name: str = "Aeris-fds-Pixcel1d-Medipix3",
    ):
        """Perform Rietveld refinement.

        Args:
            xrd_data: _description_
            cifs: _description_

        Returns
        -------
            _description_
        """
        pattern_path = Path("xrd_data.xy")
        xrd_data.to_xy_file(pattern_path)

        phase_paths = []
        for cif in cifs:
            cif_path = Path(f"{next(iter(cif.data.keys()))}.cif")
            phase_paths.append(cif_path)

            with open(cif_path, "w") as f:
                f.write(str(cif))

        args = {
            "pattern_path": pattern_path,
            "phase_paths": phase_paths,
            "instrument_name": instrument_name,
            "phase_params": phase_params,
            "refinement_params": self.refinement_params,
            "show_progress": self.show_progress,
        }

        result = do_refinement(**args) if self.save else do_refinement_no_saving(**args)

        return RefinementDocument(
            task_label=self.name,
            result=result,
            rwp=result.lst_data.rwp,
            xrd_data=xrd_data,
            input_cifs=cifs,
            refinement_params=self.refinement_params,
            phase_params=phase_params,
            instrument_name=instrument_name,
        )


@dataclass
class PhaseSearchMaker(Maker):
    """Maker to create a job for performing phase search using BGMN."""

    name: str = "phase_search"
    phase_predictor: PhasePredictor | None = field(default_factory=PhasePredictor)
    verbose: bool = True
    run_final_refinement: bool = True
    cifs_folder_name: str = "cifs"

    @job(output_schema=PhaseSearchDocument)
    def make(
        self,
        xrd_data: XRDData,
        cifs: list[Cif] | None = None,
        precursors: list[str] | None = None,
        predict_kwargs: dict | None = None,
        search_kwargs: dict | None = None,
        final_refinement_params: dict | None = None,
        computed_entries=None,
    ):
        """Perform phase search."""
        directory = Path(os.getcwd())

        if predict_kwargs is None:
            predict_kwargs = {}
        if search_kwargs is None:
            search_kwargs = {}
        if final_refinement_params is None:
            final_refinement_params = {
                "gewicht": "SPHAR4",
                "lattice_range": 0.02,
                "k1": "0_0^1",
                "k2": "0_0^0.001",
            }

        pattern_path = directory / "xrd_data.xy"
        cifs_path = directory / self.cifs_folder_name

        xrd_data.to_xy_file(pattern_path)

        if precursors is None:
            precursors = []

        if cifs is not None:
            logger.info("CIFs provided, skipping prediction.")
            for cif in cifs:
                cif_path = cifs_path / f"{next(iter(cif.data.keys()))}.cif"

                with open(cif_path, "w") as f:
                    f.write(str(cif))
        else:
            if precursors is None:
                raise ValueError("Must provide either precursors or a set of CIFs!")

            if self.phase_predictor is None:
                logger.info(
                    "Phase prediction disabled; using all ICSD phases in the chemical "
                    "system."
                )
                elems = {
                    str(elem) for p in precursors for elem in Composition(p).elements
                }
                ICSDDatabase().get_cifs_by_chemsys(elems, dest_dir=cifs_path.as_posix())
            else:
                logger.info("Predicting phases...")
                self._predict_folder(
                    precursors,
                    cifs_path=cifs_path,
                    computed_entries=computed_entries,
                    **predict_kwargs,
                )

        results = search_phases(
            pattern_path=pattern_path,
            cif_paths=list(cifs_path.glob("*.cif")),
            **search_kwargs,
        )
        self._save_results(results)

        best_result = None
        if self.run_final_refinement and results:
            logger.info("Re-refining best result...")

            for item in os.listdir(directory):
                if (
                    "1_result_rwp" in item
                    and "rwp" in item
                    and os.path.isdir(directory / item)
                ):
                    best_dir = item
                    break
            else:
                raise ValueError("Best refinement not found!")

            best_dir_path = directory / "0_best"
            if best_dir_path.exists():
                shutil.rmtree(best_dir_path)

            shutil.copytree(directory / best_dir, best_dir_path)
            shutil.copy(pattern_path, best_dir_path)
            dumpfn(final_refinement_params, best_dir_path / "phase_params.json")

            logger.info("Performing final refinement on best result...")
            best_result = do_refinement(
                pattern_path=best_dir_path / "xrd_data.xy",
                phase_paths=[p[0] for p in results[0].phases],
                phase_params=final_refinement_params,
                show_progress=True,
            )
            new_best_dir_path = (
                str(best_dir_path) + f"_rwp_{round(best_result.lst_data.rwp, 2)}"
            )

            if os.path.exists(new_best_dir_path):
                shutil.rmtree(new_best_dir_path)

            os.rename(best_dir_path, new_best_dir_path)

        parsed_results = [
            (
                [[Cif.from_file(f) for f in phase] for phase in result.phases],
                result.refinement_result,
            )
            for result in results
        ]
        all_rwp = [i[1].lst_data.rwp for i in parsed_results]

        if self.run_final_refinement and best_result is not None:
            all_rwp.append(best_result.lst_data.rwp)

        best_rwp = min(all_rwp, default=None)

        return PhaseSearchDocument(
            task_label=self.name,
            results=parsed_results,
            final_result=best_result,
            best_rwp=best_rwp,
            xrd_data=xrd_data,
            input_cifs=cifs,
            precursors=precursors,
            phase_predictor=self.phase_predictor,
            predict_kwargs=predict_kwargs,
            search_kwargs=search_kwargs,
            final_refinement_params=final_refinement_params,
            run_final_refinement=self.run_final_refinement,
            cifs_folder_name=self.cifs_folder_name,
        )

    def _predict_folder(self, precursors, cifs_path, **kwargs) -> dict:
        if self.phase_predictor is None:
            raise ValueError("Phase predictor not provided!")

        prediction = self.phase_predictor.predict(precursors, **kwargs)
        dumpfn(prediction, "prediction.json")
        cost_cutoff = 0.1

        if any("CO3" in f for f in precursors):
            logger.info(
                "Carbonate compound detected, raising cost threshold to account for "
                "incorrectly modeled thermodynamics of CO2 production."
            )
            cost_cutoff = 0.5

        self.phase_predictor.write_cifs_from_formulas(
            prediction, dest_dir=cifs_path, cost_cutoff=cost_cutoff
        )

        return prediction

    def _save_results(self, results: list[SearchResult]):
        results_sorted = sorted(results, key=lambda x: x.refinement_result.lst_data.rwp)
        for idx, search_result in enumerate(results_sorted):
            phases = search_result.phases
            main_folder = phases[0][0].parent.parent

            rwp = search_result.refinement_result.lst_data.rwp
            folder_path = main_folder / f"{idx+1}_result_rwp_{round(rwp,2)}"

            for fn in os.listdir(main_folder):
                if f"{idx+1}_result_rwp" in fn:
                    if os.path.isdir(main_folder / fn):
                        shutil.rmtree(main_folder / fn)
                    else:
                        os.remove(main_folder / fn)

                    logger.info(f"Removing old result: {fn}")

            os.mkdir(folder_path)

            for phase_i, phases_ in enumerate(phases):
                phase_folder = folder_path / f"phase_{phase_i+1}"
                phase_folder.mkdir(exist_ok=True)
                for phase in phases_:
                    shutil.copy(phase, phase_folder)

            logger.info(f"Successfully saved result {idx+1}")
