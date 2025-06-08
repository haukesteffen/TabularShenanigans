import numpy as np
from autogluon.core.metrics import make_scorer
from sklearn.metrics import root_mean_squared_log_error

def rmsle_clamped(y_true, y_pred):
    y_pred_clamped = np.maximum(y_pred, 0)
    y_true_clamped = np.maximum(y_true, 0)
    return root_mean_squared_log_error(y_true_clamped, y_pred_clamped)

ag_rmsle_clamped_scorer = make_scorer(
    name='rmsle_clamped',
    score_func=rmsle_clamped,
    optimum=0,
    greater_is_better=False
)