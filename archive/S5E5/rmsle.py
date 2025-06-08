import numpy as np
from autogluon.core.metrics import make_scorer
from sklearn.metrics import root_mean_squared_log_error

def rmsle(y_true, y_pred):
    return root_mean_squared_log_error(y_true, y_pred)

def rmsle_clamped(y_true, y_pred):
    y_pred_clamped = np.maximum(y_pred, 0)
    return root_mean_squared_log_error(y_true, y_pred_clamped)

ag_rmsle_scorer = make_scorer(
    name='rmsle',
    score_func=rmsle,
    optimum=0,
    greater_is_better=False
)

ag_rmsle_clamped_scorer = make_scorer(
    name='rmsle_clamped',
    score_func=rmsle_clamped,
    optimum=0,
    greater_is_better=False
)