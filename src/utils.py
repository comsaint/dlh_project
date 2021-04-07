import math


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
