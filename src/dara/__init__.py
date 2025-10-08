"""Automatically configures DARA settings."""

from importlib.metadata import version

from dara.refine import RefinementPhase, do_refinement, do_refinement_no_saving
from dara.search import search_phases
from dara.settings import DaraSettings

__version__ = version("dara-xrd")
SETTINGS = DaraSettings()
