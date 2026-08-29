from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, positive_scores, two_class_scores=None) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, positive_scores),
        "pr_auc": average_precision_score(y_true, positive_scores),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    if two_class_scores is not None:
        scores = np.asarray(two_class_scores, dtype=np.float64)
        one_hot = np.eye(2, dtype=np.int64)[y_true]
        result["roc_auc_legacy"] = roc_auc_score(one_hot.ravel(), scores.ravel())
    return result

