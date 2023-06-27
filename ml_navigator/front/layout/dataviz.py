from __future__ import annotations
import datetime as dt
from typing import TYPE_CHECKING

from dash import dash_table, dcc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

from ml_navigator.front.layout.utils import get_cool_palette
from ml_navigator.front.settings import PRED_COLNAME
from ml_navigator.ml.predictor import Predictor
from  ml_navigator.front.layout.utils import ORANGES

if TYPE_CHECKING: # handle circular import caused to type hinting
    from ml_navigator.front.gateway import DataSession

dviz_table_metrics = dash_table.DataTable(
    columns = [{"id": "Model", "name": "Model"}, {"id": "Metrics", "name": "Metrics"}],
    data=[],
    style_header={
        "backgroundColor": "black",
        "fontWeight": "bold",
        "border": "1px solid orange",
        "text-align": "left",
    },
    style_data={
        "backgroundColor": "black",
        "text-align": "left",
        "border": "1px solid orange",
    },
    sort_action="native",
    sort_mode="single",
    column_selectable=False,
    cell_selectable=True,
    page_action="native",
    page_current=0,
    page_size=25,
    style_data_conditional=[],
    id="dataviz-training-results-metrics")

def dviz_barchart_metrics(ds: DataSession, metric: str, sort: bool=True, ascending: bool=True):
    if sort:
        ds.df_training_results = ds.df_training_results.sort_values(metric, ascending=ascending)

    X = [model.id for model in ds.df_training_results["model_object"]]
    y = ds.df_training_results[metric]

    title = f"{metric} results by models"
    labels = {"x": "Models", "y": metric}

    figure = px.bar(ds.df_training_results, x=X, y=y, color=X, title=title, labels=labels, height=300, color_discrete_sequence=get_cool_palette(len(X)), template="plotly_dark")
    figure.update_layout(legend_title_text="", bargroupgap=0.66, bargap=0, margin={"t": 50, "b": 0}, paper_bgcolor="rgb(24,26,27)", plot_bgcolor="rgb(24,26,27)")
    figure.update_xaxes(tickangle=-35, showgrid=False, tickfont={"size": 8})
    figure.update_yaxes(showgrid=False, range=None if metric != "R2" else [0, 1])

    return figure

empty_barchart = dcc.Graph(figure=px.bar(height=300, template="plotly_dark").update_layout(paper_bgcolor="rgb(24,26,27)", plot_bgcolor="rgb(24,26,27)"), id="dataviz-barchart-metrics")



dviz_table_inferences = dash_table.DataTable(
    columns = [{"id": "Features", "name": "Features"}, {"id": "label", "name": "label"},  {"id": PRED_COLNAME, "name": PRED_COLNAME}],
    data=[],
    style_header={
        "backgroundColor": "black",
        "fontWeight": "bold",
        "border": "none",
        "text-align": "left",
    },
    style_data={
        "backgroundColor": "black",
        "text-align": "left",
    },
    sort_action="native",
    sort_mode="single",
    column_selectable=False,
    cell_selectable=True,
    page_action="native",
    page_current=0,
    page_size=25,
    id="dataviz-inferences-results")


def dviz_confusion_matrix(p: Predictor):
    y_true = list(map(str, p.y_test))
    y_pred = list(map(str, p.model.y_preds_test))
    labels = np.unique(y_true).astype(str).tolist()

    cm = confusion_matrix(y_true, y_pred).tolist()
    annotations = [[str(i) for i in j] for j in cm]
    x = labels
    y = labels

    fig = px.imshow(cm, x=x, y=y, color_continuous_scale=[[0, ORANGES[5]],[.25, ORANGES[3]],[.5, ORANGES[2]],[.75, ORANGES[1]],[1, ORANGES[0]]])

    fig.update_layout(title="Confusion Matrix", template="plotly_dark", height=450, paper_bgcolor="rgb(24,26,27)", plot_bgcolor="rgb(24,26,27)")
    fig.update_xaxes(title="Predicted Labels", side="top")
    fig.update_yaxes(title="True Labels")
    fig.update_traces(text=annotations, texttemplate="%{text}")
    fig.update_xaxes()

    return fig

def dviz_3dscatter(p: Predictor):
    X = p.X_test
    w = p.y_test.astype(str) if p.ml_pb == "Classification" else p.y_test

    pca = PCA(n_components=3, random_state=1)
    X_transformed = pca.fit_transform(X)
    x = X_transformed[:, 0]
    y = X_transformed[:, 1]
    z = X_transformed[:, 2]
    df = pd.DataFrame({"PC1": x, "PC2": y, "PC3": z, p.y_col: w})
    palette = get_cool_palette(len(set(w)))
    fig = px.scatter_3d(df, x="PC1", y="PC2", z="PC3", color=p.y_col, template="plotly_dark", height=450, color_discrete_sequence=palette, color_continuous_scale=palette)
    fig.update_layout(title="Principal Components analysis", legend_title_text=p.y_col, legend_title=p.y_col, margin=dict(l=0, r=0, b=20, t=30), paper_bgcolor="rgb(24,26,27)", plot_bgcolor="rgb(24,26,27)")

    return fig


def dviz_true_vs_pred(p: Predictor):
    y_trues = np.concatenate([p.y_train, p.y_test]).ravel()
    y_preds = np.concatenate([p.model.y_preds_train, p.model.y_preds_test]).ravel()
    split = ["Train"] * p.y_train.shape[0] + ["Test"] * p.y_test.shape[0]
    palette = get_cool_palette(2)
    df = pd.DataFrame({"True values": y_trues, "Predicted values": y_preds, "Split": split})
    fig_1 = px.scatter(df, x="True values", y="Predicted values", color="Split", template="plotly_dark", height=450, color_discrete_sequence=palette, color_continuous_scale=palette, log_x=True, log_y=True)
    fig_2 = px.line(df, x="True values", y="True values", template="plotly_dark", height=450, log_x=True, log_y=True)

    fig = go.Figure(data=fig_1.data + fig_2.data)
    fig.layout = fig_1["layout"]
    fig.update_layout(title="True vs Predicted", margin=dict(l=0, r=0, b=20, t=30), paper_bgcolor="rgb(24,26,27)", plot_bgcolor="rgb(24,26,27)")
    return fig

def dviz_timeseries(p: Predictor):
    if p.model.name != "Sarimax":
        X_test = p.model.predictor.X_test["date"].apply(dt.datetime.fromordinal)
    else:
        X_test = p.model.predictor.X_test.filter(items=["date", "dates", "DATE"]).iloc[:, 0]
    df_true = pd.DataFrame({"date": pd.to_datetime(p.model.predictor.X.filter(items=["date", "dates", "DATE"]).iloc[:, 0], format=p.date_format), "y": p.y}).sort_values("date")
    df_pred = pd.DataFrame({"date": pd.to_datetime(X_test, format=p.date_format), "y": p.model.y_preds_test}).sort_values("date")

    fig = go.Figure()
    palette = get_cool_palette(2)
    fig.add_trace(go.Scatter(
        x=df_true["date"],
        y=df_true["y"],
        marker={"color": palette[0]},
        line={"width": 4, "shape": "spline"},
        mode="lines",
        name="y true",
        # legendgroup=df_true["entity_cols"].iloc[0],
        opacity=0.8))

    # forecast data plot
    fig.add_trace(go.Scatter(
        x=df_pred["date"],
        y=df_pred["y"],
        marker={"color": palette[1]},
        line={"dash": "dot", "shape": "spline"},
        mode="lines",
        name="y pred",
        # legendgroup=df_pred["entity_cols"].iloc[0],
        opacity=1))

    fig.update_layout(title="Forecast", template="plotly_dark", paper_bgcolor="rgb(24,26,27)", plot_bgcolor="rgb(24,26,27)")
    return fig