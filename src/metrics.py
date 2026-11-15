import numpy as np
from sklearn.metrics import auc, balanced_accuracy_score, precision_recall_curve, roc_auc_score


def binary_metrics(y, prob):
    """Same definitions as the official CBraMod evaluator (finetune_evaluator.py)."""
    precision, recall, _ = precision_recall_curve(y, prob)
    return {
        "bacc": float(balanced_accuracy_score(y, prob > 0.5)),
        "aucpr": float(auc(recall, precision)),  # trapezoid PR-AUC, not average_precision_score
        "auroc": float(roc_auc_score(y, prob)),
    }


def per_subject_metrics(y, prob, subject):
    return {int(s): binary_metrics(y[subject == s], prob[subject == s]) for s in np.unique(subject)}
