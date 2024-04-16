from __future__ import annotations

import os
import re
from pathlib import Path
from subprocess import run

import pandas as pd

from dara.bgmn.download_bgmn import download_bgmn
from dara.utils import get_logger

logger = get_logger(__name__)


class ParParser:
    def __init__(self, control_file_path: Path | str):
        self.bgmn_folder = (Path(__file__).parent / "bgmn" / "BGMNwin").absolute()

        self.output_path = self.bgmn_folder / "output"

        if (
            not self.output_path.exists()
            and not self.output_path.with_suffix(".exe").exists()
        ):
            logger.warning("BGMN executable not found. Downloading BGMN.")
            download_bgmn()

        os.environ["EFLECH"] = self.bgmn_folder.as_posix()
        os.environ["PATH"] += os.pathsep + self.bgmn_folder.as_posix()

        self.control_file_path = Path(control_file_path)
        if not self.control_file_path.suffix == ".sav":
            raise ValueError("The control file has to be a .sav file.")

    def to_df(self) -> pd.DataFrame:
        """Parse the BGMN control file and return the parsed data as a DataFrame."""
        output = self._run_output()
        df = self._parse_output(output)
        df = df[df["intensity"] > 0]
        return df

    def _run_output(self) -> str:
        """Run the BGMN output executable."""
        cp = run(
            [self.output_path.as_posix(), self.control_file_path.absolute().as_posix()],
            cwd=self.control_file_path.parent.absolute().as_posix(),
            capture_output=True,
            check=True,
            timeout=60,
        )

        if cp.returncode:
            raise RuntimeError(
                f"Error in BGMN refinement for {self.control_file_path}. The exit code is {cp.returncode}\n"
                f"{cp.stdout}\n"
                f"{cp.stderr}"
            )

        return cp.stdout.decode("utf-8")

    def _parse_output(self, output: str) -> pd.DataFrame:
        """Parse the output of the BGMN refinement."""
        lines = output.split("\n")
        lines = [line for line in lines if re.search(r"^\s*\d", line) is not None][1:]
        lines = [self._parse_data_line(line) for line in lines if line.strip()]
        columns = {
            "2theta": "Float64",
            "intensity": "Float64",
            "relative_intensity": "Float64",
            "d": "Float64",
            "sk": "Float64",
            "b1": "Float64",
            "b2^2": "Float64",
            "phase": str,
            "H": "Int16",
            "h": "Int16",
            "k": "Int16",
            "l": "Int16",
            "texture": "Float64",
            "F": "Float64",
        }
        return pd.DataFrame(lines, columns=list(columns.keys())).astype(columns)

    @staticmethod
    def _parse_data_line(line: str) -> list[float | None]:
        """Parse a data line."""

        def _try_float(item: str) -> float | None | str:
            if item.strip():
                try:
                    return float(item)
                except ValueError:
                    return item
            return None

        numbers = re.split(r"\s*\|\s*", line.strip())
        return [_try_float(number) for number in numbers]
