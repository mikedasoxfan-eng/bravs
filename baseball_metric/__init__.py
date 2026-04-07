"""
BRAVS — Bayesian Runs Above Value Standard

A probabilistic baseball player valuation framework that produces
posterior distributions over player value, measured in wins above
Freely Available Talent (FAT).
"""

__version__ = "0.1.0"
__all__ = ["compute_bravs", "BRAVSResult"]

from baseball_metric.core.model import compute_bravs
from baseball_metric.core.types import BRAVSResult
