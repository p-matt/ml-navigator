from dash import html, dcc
import dash_mantine_components as dmc

from .inputs import inp_upload, inp_features, inp_date_format, inp_target, inp_ml_problematic, inp_ml_models, inp_ml_models_params, inp_auto_tune, inp_generic_button, inp_auto_tune_n_iter, inp_horizon
from .dataviz import dviz_table_metrics, empty_barchart, dviz_table_inferences

input_layout = html.Div(
    [
        html.Div(
        [
            html.H3("Input data", className="card-header"),
                html.Div(
                [
                    html.Div(
                    [
                        html.Div(
                        [
                            inp_upload("input-upload-train")
                        ], className="input-items"),

                        html.Div(
                        [
                            html.Div(
                            [
                                inp_ml_problematic
                            ], className="input-items", id="input-items-ml-pb"),

                            html.Div(
                            [
                                inp_features,
                                html.Div([inp_date_format], id="input-item-date-format")
                            ], className="input-items", id="input-items-features"),                        
                            
                            html.Div(
                            [
                                inp_target
                            ], className="input-items", id="input-items-target"),
                            
                        ], id="input-items-group1"),

                        html.Div(
                        [
                            html.Div(
                            [
                                inp_ml_models(idx) for idx in range(1,7)
                            ], className="input-items", id="input-items-ml-models"),
                            html.Div(
                            [
                                inp_ml_models_params(idx) for idx in range(1,7)
                            ], className="input-items", id="input-items-ml-models-params"),
                            

                        ], id="input-items-group2"),


                        html.Div(
                        [
                            html.Div(
                            [
                                inp_auto_tune,
                                html.Div([inp_auto_tune_n_iter], id="input-item-autotune")
                                
                            ], className="input-items", id="input-items-autotune")

                        ], id="input-items-group3"),

                        html.Div(
                        [
                            inp_generic_button("Start", "btn btn-outline-warning", "input-start-training")
                        ], className="input-items", id="input-items-start-training"),

                    
                    ], id="container-inputs-items"),
            ], className="card-body")
        ], className="card border-warning mb-3")
    ], id="container-inputs")

metrics_layout = html.Div(
    [
        html.Div(
        [
            html.H3("Training results", className="card-header"),

            html.Div(
            [
                dviz_table_metrics,
                html.Div([empty_barchart], id="dataviz-items-training-results-barchart"),
                html.Div(id="dataviz-items-training-results-additional")
            ], className="card-body")
        ], className="card border-warning mb-3"),

    ], id="container-metrics")

inferences_layout = html.Div(
    [
        html.Div(
        [
            html.H3("Inferences", className="card-header"),

            html.Div(
            [
                html.Div(
                [
                    inp_upload("input-upload-inferences")   
                ], className="input-items"),
                
                html.Div(
                [
                    html.Div(id="input-items-features-inference"),
                    dviz_table_inferences,
                ], id="output-inference"),
                html.Div(className="input-items", id="input-items-dataviz-inference"),

            ], className="card-body")
        ], className="card border-warning mb-3"),
    ], id="container-inferences")



app_layout = html.Div(
    [
        html.Nav(
        [
            html.Div(
            [
                html.H1("Machine Learning Navigator")
            ], className="container-fluid"),
        ], id="nav", className="navbar navbar-expand-lg bg-dark"),

        html.Div(
        [
            html.Div(
            [
                input_layout,
                metrics_layout,
                inferences_layout,
                html.Div(
                [
                    dcc.Loading(id="loading-1", children=[html.Div([html.Div(id="loading-output-2")])], type="circle")
                ], id="handler-loading")
                
            ], id="container-layouts"),
        ], id="container"),

        inp_generic_button("Previous", "btn btn-outline-warning", "button-previous"),
        inp_generic_button("Next", "btn btn-outline-warning", "button-next"),
        html.Div(
        [
            html.Div("No model selected - Click on the model's row you want to select", className="card-header", id="model-selected-title"),
            html.Div(
            [
                html.H5("Parameters", className="card-title"),
                html.Div(
                [
                    html.Ul(
                    [
                        
                    ], id="model-selected-params")
                ], className="card-text"),
            ], className="card-body")
        ], id="model-selected", className="card text-white border-warning mb-3"),

        
        html.Div(id="handler-01"),
        html.Div(id="handler-02"),
        html.Div(id="handler-03"),
        html.Div(id="handler-04"),
        html.Div(id="handler-05"),
        dcc.Location(id="url", refresh=False)
    ])


