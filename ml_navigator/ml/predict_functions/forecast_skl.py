from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING: # handle circular import caused to type hinting
    from ml_navigator.ml.model import Model
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', None)  # or 1000



def forecast_skl(model: Model, X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame, params_model: dict, context: Literal["evaluate", "infer"]) -> list:
    model.estimator = model._estimator_base(**params_model)
    model.estimator.fit(X_train["date"].to_numpy().reshape(-1, 1), y_train)
    if context == "evaluate":
        model.y_preds_train = model.estimator.predict(X_train["date"].to_numpy().reshape(-1, 1))
        model.y_preds_test = model.estimator.predict(X_pred["date"].to_numpy().reshape(-1, 1))
    elif context == "infer":
        model.y_preds_inference = model.estimator.predict(X_pred["date"].to_numpy().reshape(-1, 1))
        
    model.params = model.estimator.get_params()
