from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from dara.eflech_worker import EflechWorker, merge_peaks


def detect_peaks(
    pattern: Union[Path, np.ndarray],
    instrument_name: str = "Aeris-fds-Pixcel1d-Medipix3",
    wmin: float = None,
    wmax: float = None,
    resolution: float = 0.2,
) -> pd.DataFrame:
    eflech_worker = EflechWorker()
    peaks = eflech_worker.run_peak_detection(
        pattern=pattern, instrument_name=instrument_name, wmin=wmin, wmax=wmax
    )
    peaks = merge_peaks(peaks, resolution=resolution)

    return peaks
