from __future__ import annotations
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd


if TYPE_CHECKING: # handle circular import caused to type hinting
    from ml_navigator.ml.model import Model
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', None)  # or 1000



def predict_skl(model: Model, X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame, params_model: dict, context: Literal["evaluate", "infer"]) -> list:
    model.estimator = model._estimator_base(**params_model)
    model.estimator.fit(X_train, y_train.reshape(-1, 1))

    if context == "evaluate":
        model.y_preds_train = np.array(model.estimator.predict(X_train))
        model.y_preds_test = np.array(model.estimator.predict(X_pred))
    elif context == "infer":
        model.y_preds_inference = np.array(model.estimator.predict(X_pred))
   
    model.params = model.estimator.get_params()
