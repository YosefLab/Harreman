
import logging
import os
from importlib.metadata import version

from rich.console import Console
from rich.logging import RichHandler

from . import cell_communication as ccc
from . import database as db
from . import plots as pl
from . import signature as sg
from .anndata import AnnDataAccessor
from .hotspot import Hotspot
from .main import *
from .visionpy import VISION

__version__ = version("visionpy-sc")

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("visionpy: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False


data_accessor = AnnDataAccessor()
hotspot_acc = Hotspot()
vision_acc = VISION()

