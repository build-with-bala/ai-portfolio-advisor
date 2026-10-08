"""Training pipeline package (`src.data`, `src.model`).

The decision engine used by the apps lives in `src/portfolio_advisor`. Putting
this directory on the path lets the pipeline import it whether or not the
project has been pip-installed.
"""
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
