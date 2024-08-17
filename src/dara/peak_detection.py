from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from dara.eflech_worker import EflechWorker

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd


def detect_peaks(
    pattern: Path | np.ndarray,
    wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float = "Cu",
    instrument_name: str = "Aeris-fds-Pixcel1d-Medipix3",
    wmin: float = None,
    wmax: float = None,
    show_progress: bool = False,
    nthreads: int = None,
    timeout: int = 1800,
) -> pd.DataFrame:
    eflech_worker = EflechWorker()
    return eflech_worker.run_peak_detection(
        pattern=pattern,
        wavelength=wavelength,
        instrument_name=instrument_name,
        wmin=wmin,
        wmax=wmax,
        show_progress=show_progress,
        nthreads=nthreads,
        timeout=timeout,
    )
