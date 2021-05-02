import math
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from config import NUM_CLASSES, RUNS_DIR, MODEL_NAME

writer = SummaryWriter(f'{RUNS_DIR}/{MODEL_NAME}')

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

'''
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
    macro_roc_auc = auc(all_fpr, mean_tpr)
    return macro_roc_auc, roc_auc
'''


def calculate_metric(scores, truth):
    AUROCs = []
    for i in range(NUM_CLASSES):
        AUROCs.append(roc_auc_score(truth[:, i], scores[:, i]))
    macro_auroc = np.mean(np.array(AUROCs))
    return macro_auroc, AUROCs
