import warnings
import numpy as np
from sklearn.metrics import top_k_accuracy_score, make_scorer, confusion_matrix, balanced_accuracy_score

def balanced_accuracy_no_warn(y_true, y_pred):
    """balanced_accuracy_score with the 'classes not in y_true' warning suppressed.

    Defined as a named function (not a lambda) so it is picklable and can be
    shipped to joblib worker processes.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='y_pred contains classes not in y_true',
            category=UserWarning,
        )
        return balanced_accuracy_score(y_true, y_pred)


balanced_accuracy_scorer = make_scorer(balanced_accuracy_no_warn)


def per_class_metrics(y_true, y_pred):
    """Compute per-class sensitivity, specificity, PPV, and NPV from a confusion matrix.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Keys 'sensitivity', 'specificity', 'ppv', 'npv', each a numpy array
        of length n_classes.
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    sensitivity = np.zeros(n_classes)
    specificity = np.zeros(n_classes)
    ppv = np.zeros(n_classes)
    npv = np.zeros(n_classes)

    for i in range(n_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        sensitivity[i] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity[i] = TN / (FP + TN) if (FP + TN) > 0 else 0.0
        ppv[i]         = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        npv[i]         = TN / (TN + FN) if (TN + FN) > 0 else 0.0

    return {'sensitivity': sensitivity, 'specificity': specificity, 'ppv': ppv, 'npv': npv}


def weighted_top_k_accuracy_score(y_true, y_score, k=2, normalize=True, labels=None, sample_weight='frequency'):
    if sample_weight not in ['frequency']:
        raise Exception('sample_weight must be one of the following: "frequency"')
    
    if sample_weight == 'frequency':
        unique, counts = np.unique(y_true, return_counts=True)
        n_samples = counts.sum()
        class_weights = n_samples / counts
        w = class_weights[y_true]

    return top_k_accuracy_score(
        y_true=y_true, 
        y_score=y_score, 
        k=k, 
        normalize=normalize, 
        sample_weight=w, 
        labels=labels
    )