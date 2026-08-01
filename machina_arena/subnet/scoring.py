"""Machina Arena — Scoring mechanism.

Defines how miners are scored based on their policy performance.
The score determines TAO emission distribution via Yuma Consensus.

Scoring is objective and transparent:
- Average normalized dense reward over N episodes
- Success rate as a secondary metric
- Weights = softmax of scores across all miners
"""

import numpy as np


def compute_weights(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Compute weights from scores using softmax.

    Args:
        scores: List of miner scores [0, 1]
        temperature: Softmax temperature (lower = more competitive)

    Returns:
        Weights that sum to 1.0, representing emission distribution
    """
    scores_arr = np.array(scores)
    if scores_arr.sum() == 0:
        # All miners scored 0 — equal weights
        return [1.0 / len(scores)] * len(scores)

    # Softmax with temperature
    exp_scores = np.exp(scores_arr / temperature)
    weights = exp_scores / exp_scores.sum()
    return weights.tolist()


def score_to_weight(score: float, all_scores: list[float]) -> float:
    """Convert a single score to a weight relative to all miners."""
    weights = compute_weights(all_scores)
    return weights[all_scores.index(score)] if score in all_scores else 0.0
