"""
Unit tests for metric computations (Task 22: compute_eer on known inputs).
"""
import numpy as np
import pytest

from metrics import compute_eer


def test_eer_perfect_separation():
    # Labels: 50 bonafide (0), 50 spoof (1)
    labels = np.array([0] * 50 + [1] * 50)
    # Scores: bonafide get low scores (0.01 - 0.40), spoof get high scores (0.60 - 0.99)
    scores = np.concatenate([
        np.linspace(0.01, 0.40, 50),
        np.linspace(0.60, 0.99, 50)
    ])
    eer, threshold = compute_eer(labels, scores)
    assert np.isclose(eer, 0.0, atol=1e-5), f"Expected EER ~ 0.0, got {eer}"
    assert 0.40 <= threshold <= 0.60


def test_eer_identical_distributions():
    # When scores are randomly assigned independent of label, EER should be ~ 0.50
    np.random.seed(42)
    labels = np.array([0] * 500 + [1] * 500)
    scores = np.random.uniform(0.0, 1.0, 1000)
    eer, threshold = compute_eer(labels, scores)
    assert 0.45 <= eer <= 0.55, f"Expected EER ~ 0.50, got {eer}"


def test_eer_symmetric_step():
    # 4 samples: bonafide=[0.2, 0.4], spoof=[0.3, 0.5]
    # At threshold 0.35:
    # Bonafide: 0.2 < 0.35 (correct), 0.4 > 0.35 (FP) -> FPR = 1/2 = 0.5
    # Spoof: 0.3 < 0.35 (FN), 0.5 > 0.35 (correct) -> FNR = 1/2 = 0.5
    # FPR = FNR = 0.5
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.4, 0.3, 0.5])
    eer, threshold = compute_eer(labels, scores)
    assert np.isclose(eer, 0.5, atol=1e-5)


def test_eer_monotonic_return():
    # EER must always be between 0.0 and 1.0
    labels = np.random.choice([0, 1], size=100)
    scores = np.random.rand(100)
    eer, threshold = compute_eer(labels, scores)
    assert 0.0 <= eer <= 1.0
    assert np.isfinite(threshold)
