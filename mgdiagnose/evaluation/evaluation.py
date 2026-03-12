import numpy as np
from sklearn.metrics import top_k_accuracy_score

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