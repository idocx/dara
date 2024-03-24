from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from dara.eflech_worker import EflechWorker


def detect_peaks(
    pattern: Union[Path, np.ndarray],
    instrument_name: str = "Aeris-fds-Pixcel1d-Medipix3",
    wmin: float = None,
    wmax: float = None,
) -> pd.DataFrame:
    eflech_worker = EflechWorker()
    peaks = eflech_worker.run_peak_detection(
        pattern=pattern, instrument_name=instrument_name, wmin=wmin, wmax=wmax
    )

    return peaks
