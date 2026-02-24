"""
Fairness Audit Module
Detects and measures algorithmic bias in datasets
"""

from .metrics import FairnessMetrics
from .auditor import FairnessAuditor

__all__ = ['FairnessMetrics', 'FairnessAuditor']