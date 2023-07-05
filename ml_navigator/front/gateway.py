from typing import List
import os

import pandas as pd
from dash.dash_table.Format import Format, Group
from dash import dcc, html

from ml_navigator.ml.predictor import Predictor
from ml_navigator.ml.model import Model
from ml_navigator.front.layout.dataviz import dviz_3dscatter, dviz_timeseries, dviz_confusion_matrix, dviz_true_vs_pred
from ml_navigator.front.settings import PRED_COLNAME, bag_of_model_names


class DataSession:

    def __init__(self) -> None:
        from ml_navigator.ml.model import Model
        # Input data
        self.df_train: pd.DataFrame = None
        self.X_cols: List[str] = None
        self.y_col: str = None
        self.model_names: List[str] = None
        self.models_params: List[str] = 6 * [{}]
        self.auto_tune: str = "Disabled"
        self.auto_tune_n_iter: int = 100
        self.ml_pb: str = None
        self.date_format: str = "%d/%m/%Y"

        # Computed data
        self.df_training_results: pd.DataFrame = None
        self.columns_training_dviz: List[dict] = None
        self.columns_inferences_dviz: List[str] = None
        self.default_metric: str = ""
        self.model_selected: Model = None
        self.last_sorted_column: pd.Series = None
        self.df_inference: pd.DataFrame = None

        

    def set_df_metrics(self, metrics: List[str]) -> pd.DataFrame:
        columns = ["model_object", "Model"] + metrics
        self.df_training_results = pd.DataFrame(columns=columns)
        self.default_metric = "R2" if self.ml_pb in ("Regression", "Forecast") else "F1"

    def reset(self):
        self.__init__()



def compute_training_process(ds: DataSession) -> List[dict]:
    for model_name, model_param in zip(ds.model_names, ds.models_params):
        model_id = f"{model_name} {bag_of_model_names.pop()}"
        p = Predictor(ds.df_train.copy(), ds.X_cols, ds.y_col, model_name, model_param, ds.auto_tune, ds.auto_tune_n_iter, ds.ml_pb, ds.date_format, model_id)
        data_metrics = p.evaluate()
        ds.df_training_results = pd.concat([ds.df_training_results, pd.DataFrame([{"model_object": p.model, "Model": p.model.id} | data_metrics])], ignore_index=True)
        p.model.dviz_1 = dcc.Graph(figure=dviz_3dscatter(p)) if ds.ml_pb in ("Classification", "Regression") else dcc.Graph(figure=dviz_timeseries(p))
        p.model.dviz_2 = dcc.Graph(figure=dviz_confusion_matrix(p)) if ds.ml_pb == "Classification" else dcc.Graph(figure=dviz_true_vs_pred(p)) if ds.ml_pb == "Regression" else None
        p.model.dviz_params = [html.Li(f"{k}:{v}") for k,v in p.model.params.items()]
        
    ds.columns_training_dviz = [{"id": col, "name": col} for col in ds.df_training_results.columns if col != "model_object"]

    numeric_cols = list(ds.df_train[ds.X_cols].select_dtypes(include="number").columns) + [PRED_COLNAME]
    ds.columns_inferences_dviz = [{"id": col, "name": col} if col not in numeric_cols else 
                                  {
                                    "id": col,
                                    "name": col,
                                    "type": "numeric",
                                    "format": Format(
                                                group=Group.yes,
                                                groups=3,
                                                group_delimiter=" ",
                                                decimal_delimiter=",",
                                            )
                                    } 
                                    for col in ds.X_cols[:5] + [ds.y_col] + [PRED_COLNAME]]

def compute_inferences_process(ds: DataSession, df: pd.DataFrame):
    ds.df_inference = df
    ds.model_selected.predictor.infer(df)
    if ds.ml_pb == "Regression":
        y_pred = list(map(lambda x: round(x, 2), ds.model_selected.y_pred_inference))
    else:
        y_pred = ds.model_selected.y_pred_inference
    df.insert(df.shape[1], PRED_COLNAME, y_pred)