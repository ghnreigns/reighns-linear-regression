import numpy as np


def l1_loss():
    pass

def l2_loss(y_true: np.ndarray, y_pred: np.ndarray):
    """Implements L2 Loss with $l2_loss(y_true, y_pred) = \sum_{i=1}^{m} (y_true-y_pred)^2$

    Args:
        y_true (np.ndarray): [description]
        y_pred (np.ndarray): [description]

    Returns:
        [type]: [description]
    """
    l2_loss = np.sum(np.square(y_true - y_pred))
    return l2_loss

