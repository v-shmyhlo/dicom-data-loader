"""Utils"""

import numpy as np


def iou(a, b):
    """Computes IOU between 2 arrays

    :param a: array of booleans
    :param b: array of booleans
    :return: IOU
    """

    intersection = np.sum(np.logical_and(a, b))
    union = np.sum(np.logical_or(a, b))

    if union == 0.:
        return 0.
    else:
        return intersection / union
