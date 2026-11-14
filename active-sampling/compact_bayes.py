"""Compatibility entry point: compact Bayesian fitting is now always enabled."""
from ml.models.compact_bayes import CompactBayesianRidge


def activate():
    """Retained for old replay commands; models already use the compact estimator."""
    return None
