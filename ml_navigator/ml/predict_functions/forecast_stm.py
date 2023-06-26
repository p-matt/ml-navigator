from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING: # handle circular import caused to type hinting
    from ml_navigator.ml.model import Model
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', None)  # or 1000


def forecast_stm(model: Model, X_train: pd.DataFrame, y_train: np.ndarray, X_pred: pd.DataFrame, params_model: dict, context: Literal["evaluate", "infer"]) -> list:
    predict_parameters = {"steps": X_pred.shape[0]}
    if context == "evaluate":
        fit_parameters = {"disp": False}
        

        df_train = pd.DataFrame({"target": y_train})
        df_train.index = pd.Series(np.array(X_train.values).ravel())
        params_model["endog"] = df_train
        # Fit the model
        model.estimator = model._estimator_base(**params_model)
        model.estimator = model.estimator.fit(**fit_parameters)
        # Prediction of the model and update the outpout DataFrame
        sarimax_predictions = model.estimator.get_forecast(**predict_parameters).summary_frame(alpha=.8)
        model.y_preds_test = sarimax_predictions["mean"].values
        # predict_parameters = {"steps": len(train_endog)}
        # sarimax_predictions = res.get_forecast(**predict_parameters).summary_frame(alpha=.8)
        # model.y_preds_train = sarimax_predictions["mean"].values
        del params_model["endog"]
        model.params = params_model
    elif context == "infer":
        sarimax_predictions = model.estimator.get_forecast(**predict_parameters).summary_frame(alpha=.8)
        model.y_preds_inference = sarimax_predictions["mean"].values
    
    
