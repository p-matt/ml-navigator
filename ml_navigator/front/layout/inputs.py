from dash import dcc, html, dash_table
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc



inp_upload = lambda id: dcc.Upload(id=id, className="input-upload", accept=".csv", children=html.Div(["Drag and Drop or ", html.A("Select File")]))


inp_features = dmc.MultiSelect(
    id="input-features",
    data=[],
    searchable=True,
    clearable=True,
    nothingFound="No options found",
    style={},
    label="Features",
    className="dropdown")

inp_date_format = dmc.TextInput(
    id="input-date-format",
    label="Date Format",
    value="%m/%d/%Y")

inp_target = dmc.Select(
    id="input-target",
    data=[],
    searchable=True,
    clearable=True,
    nothingFound="No options found",
    style={},
    label="Target",
    className="dropdown")


inp_ml_problematic = dbc.RadioItems(
    id="input-ml-pb",
    className="btn-group",
    inputClassName="btn-check",
    labelClassName="btn btn-outline-warning",
    labelCheckedClassName="active",
    options=["Regression", "Classification", "Forecast"],
    value="")


inp_ml_models = lambda idx: dmc.Select(
    id=f"input-ml-models-{idx}",
    data=[],
    searchable=True,
    clearable=True,
    nothingFound="No options found",
    style={},
    placeholder="Model",
    # className="dropdown",
    # classNames={"mantine-Input-wrapper": "test"})
)

inp_ml_models_params = lambda id: dcc.Input(id=f"input-ml-models-params-{id}", type="text", className="form-control input-items-ml-models-params", placeholder="Model parameters", value="")


inp_auto_tune = dmc.RadioGroup(
    [dmc.Radio(label=v, value=v, color="gray") for v in ["Disabled", "Grid search", "Bayesian optim"]],
    value="Disabled",
    label="Auto tuning",
    size="sm",
    mt=10,
    id="input-autotune",
    style={"color": "white"})

inp_auto_tune_n_iter = dmc.NumberInput(
    description="Define the length of the search space",
    stepHoldDelay=500,
    stepHoldInterval=100,
    value=100,
    id="input-autotune-n-iter")

inp_generic_button = lambda label, cname, _id: html.Button(label, type="button", className=cname, id=_id)

inp_generic_radio = lambda v: dmc.Radio(label=v, value=v, color="orange")

inp_features_inference = lambda data: dmc.MultiSelect(
    id="input-features-inference",
    data=data,
    value=data[:3],
    searchable=True,
    clearable=True,
    nothingFound="No options found",
    style={},
    label="Features selected",
    className="dropdown")

inp_horizon = dmc.NumberInput(
    description="Define timeframe of prediction",
    stepHoldDelay=500,
    stepHoldInterval=1,
    value=12,
    id="input-horizon")