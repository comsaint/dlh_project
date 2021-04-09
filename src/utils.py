import math
from config import NUM_CLASSES
from sklearn.metrics import roc_curve, auc
import numpy as np


def conv_output_volume(W, F, S, P):
    """
    Given the input volume size $W$, the kernel/filter size $F$,
    the stride $S$, and the amount of zero padding $P$ used on the border,
    calculate the output volume size.
    """
    return int((W - F + 2 * P) / S) + 1


def maxpool_output_volume(W, F, S):
    """
    Given the input volume size $W$, the kernel/filter size $F$,
    the stride $S$, and the amount of zero padding $P$ used on the border,
    calculate the output volume size.
    """
    return int(math.ceil((W - F + 1) / S))


def calculate_metric(scores, truth):
    """
    calculate multi-label, macro-average AUCROC
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#multiclass-settings
    """
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(truth[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= NUM_CLASSES
    roc_auc = auc(all_fpr, mean_tpr)
    return roc_auc
