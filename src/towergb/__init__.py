"""TowerGB: Dual-tower calibrated ensemble meta-learner for tabular classification.

A zero-heavy-dependency, high-performance classifier that combines an
accuracy-focused training tower with a calibration/risk-focused tower,
unified through a Pareto arbiter for optimal ensemble weighting and
post-hoc temperature scaling for probability calibration.

Public API
----------
TowerGBClassifier : The main estimator class (scikit-learn compatible).
__version__       : Package version string.
"""

from __future__ import annotations

from towergb.estimator import TowerGBClassifier

__version__: str = "0.1.0"

__all__ = ["TowerGBClassifier", "__version__"]
