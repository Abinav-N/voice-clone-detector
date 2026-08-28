"""
Equal Error Rate (EER) — the standard ASVspoof metric.
EER is the point on the ROC curve where the false-positive rate equals the
false-negative rate; lower is better. It matters more than raw accuracy here
because spoof/bonafide classes are often imbalanced.
"""
import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(labels: np.ndarray, scores: np.ndarray):
    """
    labels: 0 = bonafide (real), 1 = spoof (fake)
    scores: model's P(spoof) for each sample, higher = more likely fake

    Returns (eer, threshold_at_eer)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    # find the threshold where FPR and FNR are closest
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer, thresholds[idx]
