from __future__ import annotations # handle circular import caused to type hinting
import itertools
from typing import TYPE_CHECKING, Literal, Union
import itertools as it
from joblib import Parallel, delayed
from functools import partial

from tqdm_joblib import tqdm_joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, f1_score
from numpy import float64
import numpy.typing as npt
import numpy as np
from skopt.space import Integer, Real, Categorical, Space
from skopt import gp_minimize
from skopt.utils import use_named_args

if TYPE_CHECKING: # handle circular import caused to type hinting
    from .predictor import Predictor
    from .model import Model

MODELS_PARAMS_SEARCH_SPACE = {
    "Regression": {
        "Linear Regression": 
        {
            "fit_intercept": {"type": "Categorical", "values": [True, False]}
        },
        "Stochastic Gradient Descent": 
        {
            "tol": {"type": "Real", "min": .0001, "max": .01},
            "eta0": {"type": "Real", "min": .001, "max": .1},
        },
        "Random Forest": 
        {
            "n_estimators": {"type": "Integer", "min": 50, "max": 1000}
        },
        "Gradient Boosting": 
        {
            "n_estimators": {"type": "Integer", "min": 50, "max": 1000},
            "max_depth": {"type": "Integer", "min": 3, "max": 9},
            "subsample": {"type": "Real", "min": 0.5, "max": 1},
            "min_samples_split": {"type": "Integer", "min": 2, "max": 5},
            "learning_rate": {"type": "Real", "min": 0.0001, "max": 0.1}
        },
        "Support Vector Machines": 
        {
            "kernel": {"type": "Categorical", "values": ["linear", "rbf", "poly"]},
        },
    },
    "Classification": {

        "Logistic Regression": 
        {
            "solver": {"type": "Categorical", "values": ["lbfgs", "liblinear"]},
        },
        "Stochastic Gradient Descent": 
        {
            "learning_rate": {"type": "Categorical", "values": ["optimal", "adaptive"]},
            "eta0": {"type": "Real", "min": 0.0003, "max": 0.03}
        },
        "Random Forest": 
        {
            "n_estimators": {"type": "Integer", "min": 50, "max": 1000},
            "min_samples_split": {"type": "Integer", "min": 2, "max": 5},
        },
        "Gradient Boosting":
        {
            "n_estimators": {"type": "Integer", "min": 50, "max": 1000},
            "learning_rate": {"type": "Real", "min": 0.0001, "max": 0.1},
            "max_depth": {"type": "Integer", "min": 3, "max": 9},
        },
        "Support Vector Machines":
        {
            "kernel": {"type": "Categorical", "values": ["linear", "rbf", "poly"]}
        },
        "KNN":
        {
            "n_neighbors": {"type": "Integer", "min": 3, "max": 10},
            "algorithm": {"type": "Categorical", "values": ["ball_tree", "kd_tree"]}
        }
    },
    "Forecast": {
        "Sarimax": 
        {
            "initialization": {"type": "Categorical", "values": ["approximate_diffuse"]},
            "seasonal_order": {"type": "Categorical", "values": [x for x in itertools.product([0, 1], [0, 1], [0, 1], [12])]}
        }, 
        "Linear Regression": 
        {
            "fit_intercept": {"type": "Categorical", "values": [True, False]}
        },
        "Random Forest": 
        {
            "n_estimators": {"type": "Integer", "min": 50, "max": 1000}
        },
        "Gradient Boosting": 
        {
            "n_estimators": {"type": "Integer", "min": 50, "max": 1000},
            "max_depth": {"type": "Integer", "min": 3, "max": 9},
            # "subsample": {"type": "Real", "min": 0.5, "max": 1},
            # "min_samples_split": {"type": "Integer", "min": 2, "max": 5},
            # "learning_rate": {"type": "Real", "min": 0.0001, "max": 0.1}
        },
    }
}


MANDATORY_PARAMS = {
    "Regression": 
    {
        "Linear Regression": {"n_jobs": -1},
        "Stochastic Gradient Descent": {"random_state": 1},
        "Random Forest": {"n_estimators": 150, "n_jobs": -1, "random_state": 1},
        "Gradient Boosting": {"n_estimators": 150, "random_state": 1},
        "Support Vector Machines": {}
    },
    "Classification": 
    {
        "Logistic Regression": {"n_jobs": -1, "random_state": 1},
        "Stochastic Gradient Descent": {"n_jobs": -1, "random_state": 1},
        "Random Forest": {"n_estimators": 150, "random_state": 1},
        "Gradient Boosting": {"n_estimators": 150, "random_state": 1},
        "Support Vector Machines": {"random_state": 1},
        "KNN": {"n_jobs": -1}
    },
    "Forecast": 
    {
        "Sarimax": {"initialization": "approximate_diffuse", "seasonal_order": [1, 1, 1, 12]},
        "Linear Regression": {"n_jobs": -1},
        "Random Forest": {"n_estimators": 150, "n_jobs": -1, "random_state": 1},
        "Gradient Boosting": {"n_estimators": 150, "random_state": 1},
    }
}

def compute_model_parameters(predictor: Predictor):
    method = "grid search" if predictor.auto_tune == "Grid search" else "bayesian" if predictor.auto_tune == "Bayesian optim" else ""
    parameters = optimize(predictor, method) if method else predictor.model.params_user

    for key, value in predictor.model.params_mandatory.items():
        if not key in parameters:
            parameters[key] = value

    predictor.model.params = parameters.copy()

def get_grid_length(predictor: Predictor) -> int:
    return int(np.exp(np.log(predictor.auto_tune_n_iter) / len(predictor.model.params_search_space))) + 1

def get_search_space(predictor: Predictor, method: Literal["grid search", "bayesian"]) -> dict | list:
    grid_len = get_grid_length(predictor)
    search_space = [] if method == "bayesian" else {}
    for key in predictor.model.params_search_space.keys():
        _min = predictor.model.params_search_space[key].get("min", 0)
        _max = predictor.model.params_search_space[key].get("max", 1)
        _type = predictor.model.params_search_space[key]["type"]
        
        if _type in ("Real", "Integer"):
            if method == "grid search":
                search_space[key] = list(np.linspace(_min, _max, grid_len))
                if _type == "Integer":
                    search_space[key] = list(map(int, search_space[key]))
            elif method == "bayesian":
                if _type == "Integer":
                    search_space.append(Integer(_min, _max, name=key))
                else:
                    search_space.append(Real(_min, _max, "log-uniform", name=key))
        elif _type == "Categorical":
            _values = predictor.model.params_search_space[key]["values"]
            if method == "grid search":
                search_space[key] = _values
            elif method == "bayesian":
                # if isinstance(_values, list):
                #     sub_search_space = []
                #     for i, _sub_values in enumerate(_values):
                #         sub_search_space.append(Categorical(_sub_values, name=f"{key}_{i}"))
                #     search_space.append(sub_search_space)
                # else: TODO
                search_space.append(Categorical(_values, name=key))

    return search_space

def optimize(predictor: Predictor, method: Literal["grid search", "bayesian"]) -> dict:
    metrics_params = {"y_true": predictor.y_test}
    if predictor.ml_pb in ("Regression", "Forecast"):
        metrics_function = r2_score
    else:
        metrics_function = f1_score
        y_unique, y_counts = np.unique(predictor.y_test, return_counts=True)
        avg = "binary" if y_unique.shape[0] == 2 else "weighted"
        pos_label = y_unique[np.argsort(y_counts)][0] if avg == "binary" else 1
        metrics_params["average"] = avg
        metrics_params["pos_label"] = pos_label

    search_space = get_search_space(predictor, method)
    model_params = list(combinations(search_space))
    if method == "grid search":
        with tqdm_joblib(desc="Progression de l'autotuning", total=len(model_params)) as progress:
            metrics_models = Parallel(backend="multiprocessing", n_jobs=-1)(delayed(evaluate_model)(predictor.model, predictor.X_train, predictor.y_train, predictor.X_test, metrics_function, metrics_params, mp) for mp in model_params)
            best_params = model_params[metrics_models.index(max(metrics_models))]
    elif method == "bayesian":
        @use_named_args(search_space)
        def evaluate_model_bayesian(**params: dict) -> float:
            predictor.model.predict(predictor.model, predictor.X_train, predictor.y_train, predictor.X_test, params, "evaluate")
            metrics_params["y_pred"] = predictor.model.y_preds_test
            metric = metrics_function(**metrics_params)
            return -metric
        
        print(search_space)
        results = gp_minimize(evaluate_model_bayesian, search_space)
        best_params = {k: v for k,v in zip(predictor.model.params_search_space.keys(), results.x)}

    return best_params

def combinations(params_grid):
    '''The function generates all possible combinations of parameters in a nested
    dictionary or list.
    
    Parameters
    ----------
    params_grid
        params_grid is a dictionary or a list of dictionaries that contains the
    parameters and their possible values for a machine learning model. The function
    combinations generates all possible combinations of these parameters for
    hyperparameter tuning.
    
    '''
    if isinstance(params_grid, list):
        for i in params_grid:
            yield from ([i] if not isinstance(i, (dict, list)) else combinations(i))
    else:
        for i in it.product(*map(combinations, params_grid.values())):
            yield dict(zip(params_grid.keys(), i))


def evaluate_model(model: Model, X_train_scaled: np.ndarray, y_train: np.ndarray, X_test_scaled: np.ndarray, metrics_function: function, metrics_params: dict, model_params) -> float:
    """
    evaluate r2 or f1score of a model
    """
    model.predict(model, X_train_scaled, y_train, X_test_scaled, model_params, "evaluate")
    metrics_params["y_pred"] = model.y_preds_test
    metric = metrics_function(**metrics_params)
    return metric

