from __future__ import annotations # handle circular import caused to type hinting
from typing import TYPE_CHECKING

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression, SGDClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm
import numpy as np

from ml_navigator.ml.predict_functions.predict_skl import predict_skl
from ml_navigator.ml.predict_functions.forecast_skl import forecast_skl
from ml_navigator.ml.predict_functions.forecast_stm import forecast_stm
from ml_navigator.ml.params import MANDATORY_PARAMS, MODELS_PARAMS_SEARCH_SPACE


if TYPE_CHECKING: # handle circular import caused to type hinting
    from .predictor import Predictor

PREDICT_FUNCTIONS = {
    "Regression": {
        "Linear Regression": predict_skl,
        "Stochastic Gradient Descent": predict_skl,
        "Random Forest": predict_skl,
        "Gradient Boosting": predict_skl,
        "Support Vector Machines": predict_skl,
    },
    "Classification": {

        "Logistic Regression": predict_skl,
        "Stochastic Gradient Descent": predict_skl,
        "Random Forest": predict_skl,
        "Gradient Boosting": predict_skl,
        "Support Vector Machines": predict_skl,
        "KNN": predict_skl,
    },
    "Forecast": {
        "Sarimax": forecast_stm,
        "Linear Regression": forecast_skl,
        "Random Forest": forecast_skl,
        "Gradient Boosting": forecast_skl,
    }
}

MODELS = {
    "Regression": {
        "Linear Regression": LinearRegression,
        "Stochastic Gradient Descent": SGDRegressor,
        "Random Forest": RandomForestRegressor,
        "Gradient Boosting": GradientBoostingRegressor,
        "Support Vector Machines": SVR,
    },
    "Classification": {
        "Logistic Regression": LogisticRegression,
        "Stochastic Gradient Descent": SGDClassifier,
        "Random Forest": RandomForestClassifier,
        "Gradient Boosting": GradientBoostingClassifier,
        "Support Vector Machines": SVC,
        "KNN": KNeighborsClassifier
    },
    "Forecast": {
        "Sarimax": sm.tsa.SARIMAX,
        "Linear Regression": LinearRegression,
        "Random Forest": RandomForestRegressor,
        "Gradient Boosting": GradientBoostingRegressor,
    }
}


class Model:

    def __init__(self, name: str, params_user: dict, ml_type: str, predictor: Predictor, auto_tune: str, id: str) -> None:
        self.name = name
        self.params_user = params_user.copy()
        self.predictor = predictor
        self.id = id
        self.auto_tune = auto_tune
        self._estimator_base = MODELS[ml_type][name]
        self.estimator = None
        self.predict: function = PREDICT_FUNCTIONS[ml_type][name]
        
        self.params_mandatory: dict = MANDATORY_PARAMS[ml_type][name]
        self.params_search_space: dict = MODELS_PARAMS_SEARCH_SPACE[ml_type][name]
        self.params: dict = {}

        self.y_preds_train: np.array = None
        self.y_preds_test: np.array = None
        self.y_preds_inference: np.array = None

        self.dviz_1 = None
        self.dviz_2 = None
        self.dviz_params: list = None
        

