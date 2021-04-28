import math
import numpy as np
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from config import NUM_CLASSES, WRITER_NAME, LEARNING_RATE, VERBOSE
from torch.optim.lr_scheduler import ReduceLROnPlateau

writer = SummaryWriter(WRITER_NAME)


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
    AUROCs = []
    for i in range(NUM_CLASSES):
        AUROCs.append(roc_auc_score(truth[:, i], scores[:, i]))
    macro_auroc = np.mean(np.array(AUROCs))
    return macro_auroc, AUROCs


def make_optimizer_and_scheduler(m, lr=LEARNING_RATE):
    # Optimizer
    # to unfreeze certain trainable layers, use: `optimizer.add_param_group({'params': model.<layer>.parameters()})`
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=lr, weight_decay=1e-2)
    # wrap by scheduler
    # switch the mode between 'min' and 'max' depending on the metric
    # e.g. 'min' for loss (less is better), 'max' for AUC (greater is better)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=VERBOSE)
    return optimizer, scheduler
