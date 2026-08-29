"""TowerGB package.

This package helps predict categories for data in tables.
"""

from __future__ import annotations

from towergb.estimator import TowerGBClassifier

# Version of the package
__version__: str = "0.1.0"

# Things you can import from this package
__all__ = ["TowerGBClassifier", "__version__"]
