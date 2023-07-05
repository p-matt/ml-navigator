import json
from typing import List, Literal
from functools import partial

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from darts.utils.statistics import check_seasonality
from darts.timeseries import TimeSeries

from ml_navigator.ml.model import Model
from ml_navigator.ml.params import compute_model_parameters
from .utils import get_unique_values

pd.options.mode.chained_assignment = None  # default="warn"
pd.set_option("display.max_rows", None)  # or 1000
pd.set_option("display.max_columns", 14)
pd.set_option("display.width", 1300)

class Predictor():
    
    def __init__(self, df: pd.DataFrame, X_cols: List[str], y_col: str, model_name: str, model_params_user: dict, auto_tune: str, auto_tune_n_iter: int, ml_pb: str, date_format: str, id: str):
        self.X_cols = X_cols
        self.y_col = y_col
        self.model = Model(model_name, model_params_user, ml_pb, self, auto_tune, id)
        self.auto_tune = auto_tune
        self.auto_tune_n_iter = auto_tune_n_iter
        self.ml_pb = ml_pb
        self.data = df.sample(frac=1, random_state=1) if ml_pb != "Forecast" else df
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=.33, random_state=1, shuffle=True if ml_pb != "Forecast" else False)
        self.X_infer: np.ndarray = None
        self.date_format = date_format
        # self.seasonality: tuple[bool, int] = None

        np.random.seed(1)


    @property
    def X(self) -> pd.DataFrame:
        return self.data[self.X_cols]
    
    @property
    def y(self) -> np.array:
        return self.data[self.y_col].to_numpy()
    
    def evaluate(self) -> dict:
        print(f"Evaluation of {self.model.id}")
        context = "evaluate"
        self.preprocessing(context)
        compute_model_parameters(self)
        self.model.predict(self.model, self.X_train, self.y_train, self.X_test, self.model.params, context)
        metrics = self.get_metrics()
        return metrics

    def infer(self, df_inference: pd.DataFrame=None):
        print(f"Inferences with {self.model.id}")
        context = "infer"
        self.X_infer = df_inference[self.X_cols]
        self.preprocessing(context)
        self.model.predict(self.model, self.X_train, self.y, self.X_infer, self.model.params, context)

    
    def preprocessing(self, context: Literal["evaluate", "infer"]) -> Pipeline:
        numeric_columns = self.X.select_dtypes(include="number").columns
        categoric_features = self.X.select_dtypes(exclude="number")

        main_condition = (categoric_features.nunique() > 10)

        ordinal_columns = categoric_features[main_condition[main_condition].index].columns
        ohe_columns = categoric_features[main_condition[~main_condition].index].columns

        if context == "evaluate":
            categories_ordinal = get_unique_values(self.X[ordinal_columns])
            categories_ohe = get_unique_values(self.X[ohe_columns])
        else:
            categories_ordinal = get_unique_values(self.X[ordinal_columns], self.X_infer[ordinal_columns])
            categories_ohe = get_unique_values(self.X[ohe_columns], self.X_infer[ohe_columns])

        numeric_transform = Pipeline(
            steps = [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )
        ordinal_transform = Pipeline(
            steps = [
                ("ordinal_encoder", OrdinalEncoder(categories=categories_ordinal)),
            ]
        )

        ohe_transform = Pipeline(
            steps = [
                ("ohe_encoder", OneHotEncoder(categories=categories_ohe)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers = [
                ("numeric", numeric_transform, numeric_columns),
                ("cat_ord", ordinal_transform, ordinal_columns),
                ("cat_ohe", ohe_transform, ohe_columns)
            ]
        )

        X_train_preprocessed, X_test_preprocessed = (self.X_train.copy(), self.X_test.copy()) if context == "evaluate" else (self.X.copy(), self.X_infer.copy())

        if self.ml_pb == "Forecast":
            X_train_preprocessed = X_train_preprocessed.filter(items=["date", "dates", "DATE"])
            X_test_preprocessed = X_test_preprocessed.filter(items=["date", "dates", "DATE"])

            X_train_preprocessed = X_train_preprocessed.rename(columns={X_train_preprocessed.columns[0]: "date"})
            X_test_preprocessed = X_test_preprocessed.rename(columns={X_test_preprocessed.columns[0]: "date"})

            X_train_preprocessed = X_train_preprocessed[["date"]]
            X_test_preprocessed = X_test_preprocessed[["date"]]

            X_train_preprocessed = X_train_preprocessed.apply(partial(pd.to_datetime, format=self.date_format))
            X_test_preprocessed = X_test_preprocessed.apply(partial(pd.to_datetime, format=self.date_format))

            # print(self.X_train["date"])
            # train = pd.DataFrame({"date": self.X_train["date"], "y": self.y_train})
            # ts = TimeSeries.from_dataframe(train, time_col="date", value_cols="y", freq=None)
            # print(ts.freq)
            # self.seasonality = check_seasonality()
            # print(self.seasonality)
            # fft = np.fft.fft(self.X_train)
            # sampling_rate = 1  # Taux d'échantillonnage (par exemple, 1 pour des données échantillonnées à intervalles réguliers)
            # n = len(self.X_train)  # Nombre total d'échantillons
            # frequencies = np.fft.fftfreq(n, 1/sampling_rate)
            # max_index = np.argmax(np.abs(fft[1:n//2])) + 1
            # dominant_frequency = frequencies[max_index]
            # print(dominant_frequency)
            if self.model.name != "Sarimax":
                X_train_preprocessed["date"] = X_train_preprocessed["date"].apply(lambda x: x.toordinal())
                X_test_preprocessed["date"] = X_test_preprocessed["date"].apply(lambda x: x.toordinal())
        else:
            X_train_preprocessed = preprocessor.fit_transform(X_train_preprocessed)
            X_test_preprocessed = preprocessor.transform(X_test_preprocessed)
        if context == "evaluate":
            self.X_train = X_train_preprocessed
            self.X_test = X_test_preprocessed
        else:
            self.X_train = X_train_preprocessed
            self.X_infer = X_test_preprocessed
        return preprocessor
    
    def get_metrics(self) -> dict:
        y_unique, y_counts = np.unique(self.y_test, return_counts=True)
        avg = "binary" if y_unique.shape[0] == 2 else "weighted"
        
        pos_label = y_unique[np.argsort(y_counts)][0] if avg == "binary" else 1

        if self.ml_pb in ("Regression", "Forecast"):
            data_metrics = {
                "MAPE": round(mean_absolute_percentage_error(self.y_test, self.model.y_pred_test), 2),
                "MAE": round(mean_absolute_error(self.y_test, self.model.y_pred_test), 2),
                "RMSE": round(mean_squared_error(self.y_test, self.model.y_pred_test, squared=False), 2),
                "R2": round(r2_score(self.y_test, self.model.y_pred_test), 2)
            }
        elif self.ml_pb == "Classification":
             data_metrics = {
                "F1": round(f1_score(self.y_test, self.model.y_pred_test, average=avg, pos_label=pos_label), 2),
                "Precision": round(precision_score(self.y_test, self.model.y_pred_test, average=avg, pos_label=pos_label), 2),
                "Recall": round(recall_score(self.y_test, self.model.y_pred_test, average=avg, pos_label=pos_label), 2),
                "Accuracy": round(accuracy_score(self.y_test, self.model.y_pred_test), 2)
            }
        return data_metrics