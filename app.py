import os

from dash import Dash
import dash_bootstrap_components as dbc

from ml_navigator.front.layout.layouts import app_layout
from ml_navigator.front.callbacks import listen_callbacks
from ml_navigator.front.gateway import DataSession


def init_application() -> Dash:
    external_stylesheets = [
        dbc.themes.DARKLY
    ]
    
    app = Dash(__name__,
               external_stylesheets=external_stylesheets,
               assets_folder=os.path.join(os.getcwd(), "ml_navigator", "front", "assets"),
               suppress_callback_exceptions=True,
               prevent_initial_callbacks="initial_duplicate")
    app.title = "ML Navigator"
    app.layout = app_layout
    return app


# Creating a Dash app and setting the layout to the app_layout function.
if __name__ == "__main__":
    app = init_application()
    dev = False
    if dev:
        import pickle
        with open("datasession.pickle", "rb") as inp:
            ds = pickle.load(inp)
    else:
        ds = DataSession()
    listen_callbacks(app, ds)
    app.run(debug=True, port=8052)
