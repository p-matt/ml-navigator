from __future__ import annotations
from typing import TYPE_CHECKING
import json

from dash import Dash, no_update, dcc
from dash.dependencies import Input, Output, State

from .layout.utils import display_item, hide_item, parse_content, highlight_bar, new_style_inferences_table
from .gateway import DataSession, compute_training_process, compute_inferences_process
from .layout.dataviz import dviz_barchart_metrics
from .layout.inputs import inp_features_inference, inp_generic_radio
from ml_navigator.ml.params import MANDATORY_PARAMS
from ml_navigator.front.settings import PRED_COLNAME

if TYPE_CHECKING:
    from ml_navigator.front.gateway import DataSession
    
def listen_callbacks(app: Dash, ds: DataSession):

    @app.callback(Output("handler-01", "style"),
                  Input("url", "pathname"))
    def on_page_reload(url):
        # ds.reset()
        pass

    @app.callback(Output("input-items-ml-pb", "style"),
                  Output("input-features", "data"),
                  Input("input-upload-train", "contents"))
    def on_file_load_training(content):
        if not content:
            return no_update  
        
        errors, df = parse_content(content)
        if errors:
            return no_update 
        
        ds.df_train = df

        return display_item, list(df.columns)
          
    

    @app.callback(Output("input-items-features", "style"),
                  Output("input-ml-models-1", "data"),
                  Output("input-ml-models-2", "data"),
                  Output("input-ml-models-3", "data"),
                  Output("input-ml-models-4", "data"),
                  Output("input-ml-models-5", "data"),
                  Output("input-ml-models-6", "data"),
                  Output("dataviz-items-training-results-additional", "style"),
                  Output("input-autotune", "children"),
                  #Output("dataviz-items-training-results-additional", "children", allow_duplicate=True),
                  Output("input-item-date-format", "style"),
                  Input("input-ml-pb", "value"))
    def on_click_ml_problematic(ml_pb):
        if not ml_pb:
            return no_update 
        style_date_format = {"display": "none"}
        style_additional_fig = {"display": "flex"}
        output_autotune = [inp_generic_radio("Disabled"), inp_generic_radio("Grid search"), inp_generic_radio("Bayesian optim")]
        ds.ml_pb = ml_pb
        if ds.ml_pb == "Forecast":
            metrics = ["MAE", "RMSE", "R2"]
            style_date_format = {"display": "unset"}
            style_additional_fig = {"display": "unset"}
            output_autotune = output_autotune[:-1]
        else:
            if ds.ml_pb == "Regression":
                metrics = ["MAE", "RMSE", "R2"]
            else:
                metrics = ["F1", "Precision", "Recall", "Accuracy"]

        ds.set_df_metrics(metrics)
        models = list(MANDATORY_PARAMS[ds.ml_pb].keys())

        return display_item, *[models] * 6, style_additional_fig, output_autotune, style_date_format
        
         
    

    @app.callback(Output("input-items-target", "style"),
                  Output("input-target", "data"),
                  Input("input-features", "value"))
    def on_select_features(features):
        if not features:
            return no_update
        
        ds.X_cols = features
        new_targets_selection = [col for col in ds.df_train.columns if col not in features]

        return display_item, new_targets_selection
        
        
    @app.callback(Output("handler-05", "style"),
                  Input("input-date-format", "value"))
    def on_select_date_format(value):
        if not value:
            return no_update
        
        ds.date_format = value
        return no_update

    @app.callback(Output("input-items-ml-models", "style"),
                  Input("input-target", "value"))
    def on_select_target(target):
        if not target:
            return no_update
        
        ds.y_col = target

        return display_item
        
        
    @app.callback(Output("input-ml-models-params-1", "style"),
                  Output("input-ml-models-params-2", "style"),
                  Output("input-ml-models-params-3", "style"),
                  Output("input-ml-models-params-4", "style"),
                  Output("input-ml-models-params-5", "style"),
                  Output("input-ml-models-params-6", "style"),
                  Output("input-ml-models-params-1", "value"),
                  Output("input-ml-models-params-2", "value"),
                  Output("input-ml-models-params-3", "value"),
                  Output("input-ml-models-params-4", "value"),
                  Output("input-ml-models-params-5", "value"),
                  Output("input-ml-models-params-6", "value"),
                  Output("input-items-autotune", "style"),
                  Output("input-items-start-training", "style"),
                  Input("input-ml-models-1", "value"),
                  Input("input-ml-models-2", "value"),
                  Input("input-ml-models-3", "value"),
                  Input("input-ml-models-4", "value"),
                  Input("input-ml-models-5", "value"),
                  Input("input-ml-models-6", "value"))
    def on_select_ml_models(*args):
        if not any(args):
            return no_update
        
        ds.model_names = [model for model in args if model]
        output_next = [display_item] * 2
        output_params_state = [display_item if model else hide_item for model in args]
        output_params_value = [f"{MANDATORY_PARAMS[ds.ml_pb][model]}".replace("'", '"') if model else "{}" for model in args]
        return output_params_state + output_params_value + output_next 
        
        
    
    
    @app.callback(Output("handler-02", "style"),
                  Input("input-ml-models-params-1", "value"),
                  Input("input-ml-models-params-2", "value"),
                  Input("input-ml-models-params-3", "value"),
                  Input("input-ml-models-params-4", "value"),
                  Input("input-ml-models-params-5", "value"),
                  Input("input-ml-models-params-6", "value"))
    def on_select_ml_models_params(*ml_models_params):
        for i, ml_model_param in enumerate(ml_models_params):
            if ml_model_param:
                try:
                    ds.models_params[i] = json.loads(ml_model_param.replace("'", '"'))
                except Exception as e:
                    print(f"wrong models' arguments \n {e}")

        return no_update
    
    @app.callback(Output("input-item-autotune", "style"),
                  Input("input-autotune", "value"))
    def on_click_auto_tune(auto_tune):
        ds.auto_tune = auto_tune
        autotune_style = hide_item if not auto_tune or auto_tune in ("disabled", "bayesian optim") else display_item
        return autotune_style
    
    @app.callback(Output("handler-03", "style"),
                Input("input-autotune-n-iter", "value"))
    def on_click_search_space_length(n):
        ds.auto_tune_n_iter = n

        return no_update
    

    @app.callback(
        Output("dataviz-barchart-metrics", "figure"),
        Output("dataviz-training-results-metrics", "style_data_conditional"),
        Output("model-selected-title", "children"),
        Output("model-selected-params", "children"),
        Output("dataviz-items-training-results-additional", "children"),
        Output("input-upload-inferences", "style"),
        Input("dataviz-training-results-metrics", "sort_by"),
        Input("dataviz-training-results-metrics", "active_cell"),
        State("dataviz-barchart-metrics", "figure"), allow_duplicate=True
    )
    def on_sort_metrics_or_select_trained_model(sorted_columns, cell_selected, figure):
        output_main_figure = output_style = output_model_selected = output_model_selected_params = output_figure_additionals = style_inp1_inference = no_update
        if sorted_columns and (ds.last_sorted_column != sorted_columns[0]):
            ds.last_sorted_column = sorted_columns[0]
            metric = ds.last_sorted_column["column_id"]
            ascending = (ds.last_sorted_column["direction"] == "asc")
            output_main_figure = dviz_barchart_metrics(ds, metric, ascending=ascending)
            if ds.model_selected:
                output_style = [{
                    "if": {"row_index": ds.df_training_results.index.get_loc(ds.df_training_results.loc[ds.df_training_results["model_object"] == ds.model_selected].index[0])},
                    "backgroundColor": "#786947",
                }]
        elif cell_selected:
            # Selection du modèle par l'utilisateur
            # Calcul et affichage des viz
            output_style = [{
                "if": {"row_index": cell_selected["row"]},
                "backgroundColor": "#786947",
            }]
            try:
                ds.model_selected = ds.df_training_results.iloc[cell_selected["row"]]["model_object"]
            except IndexError:
                print("Index error")
                return no_update
            output_main_figure = figure
            output_model_selected = f"{ds.model_selected.id} - Autotune: {ds.model_selected.auto_tune}"
            output_model_selected_params = ds.model_selected.dviz_params

            output_figure_additionals = [ds.model_selected.dviz_1, ds.model_selected.dviz_2]
            style_inp1_inference = display_item
        else:
            return no_update
        
        highlight_bar(ds.model_selected, output_main_figure)

        return output_main_figure, output_style, output_model_selected, output_model_selected_params, output_figure_additionals, style_inp1_inference

    @app.callback(Output("dataviz-items-training-results-barchart", "children"),
                  Output("dataviz-training-results-metrics", "data"),
                  Output("dataviz-training-results-metrics", "columns"),
                  Output("dataviz-inferences-results", "columns", allow_duplicate=True),
                  Output("loading-1", "children"),
                  Input("input-start-training", "n_clicks"))
    def on_click_start_training(n_clicks):
        if not n_clicks:
            return no_update
        with open("datasession.pickle", "wb") as handle:
            import pickle
            pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        compute_training_process(ds)
        out = (
            dcc.Graph(figure=dviz_barchart_metrics(ds, ds.default_metric, ascending=False), id="dataviz-barchart-metrics"),
            ds.df_training_results.drop("model_object", axis=1).to_dict(orient="records"),
            ds.columns_training_dviz, 
            ds.columns_inferences_dviz,
            no_update)
        return out



    @app.callback(Output("input-items-features-inference", "children"),
                  Output("dataviz-inferences-results", "style_data_conditional"),
                  Output("dataviz-inferences-results", "data", allow_duplicate=True),
                  Input("input-upload-inferences", "contents"))
    def on_inferences(content):
        if not all((ds.model_selected, content)):
            return no_update

        errors, df = parse_content(content)
        if errors:
            print(errors)
            return no_update
        
        compute_inferences_process(ds, df)

        style_inferences_table = new_style_inferences_table(ds) if ds.ml_pb == "Classification" else no_update
        print("Inferences done")
        return inp_features_inference(ds.model_selected.predictor.X_cols), style_inferences_table, ds.df_inference.to_dict(orient="records")
        
         
    @app.callback(Output("dataviz-inferences-results", "columns", allow_duplicate=True),
                  Output("dataviz-inferences-results", "data", allow_duplicate=True),
                  Input("input-features-inference", "value"))
    def on_select_inference_features(features):
        ds.columns_inferences_dviz = [{"id": col, "name": col} for col in features + [ds.y_col, PRED_COLNAME]]
        return ds.columns_inferences_dviz, ds.df_inference[features + [ds.y_col, PRED_COLNAME]].to_dict(orient="records")