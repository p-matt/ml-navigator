from __future__ import annotations
import base64
from io import StringIO
from typing import TYPE_CHECKING

from pandas import read_csv
import numpy as np
from plotly.graph_objects import Figure

from ml_navigator.ml.model import Model
from ml_navigator.front.settings import PRED_COLNAME

if TYPE_CHECKING:
    from ml_navigator.front.gateway import DataSession

display_item = {"visibility": "visible"}
hide_item = {"visibility": "hidden"}

ORANGES = ["#f7a23d", "#f8b76d", "#fac29c", "#fbd9cb", "#E65D0E", "#f39c13"][::-1]







def parse_content(content: str):
    '''This function takes in a string of encoded CSV content, decodes it, and attempts to read it as a CSV
    file, returning any errors encountered and the resulting DataFrame.
    
    Parameters
    ----------
    content : str
        The `content` parameter is a string that contains a comma-separated value (CSV) file encoded in
    base64 format.
    
    Returns
    -------
        The function `parse_content` returns a tuple containing an error (if any) and a pandas DataFrame
    (if successfully parsed).
    
    '''
    error = None
    df = None
    content_type, content = content.split(",")
    _bytes = base64.b64decode(content)
    try:
        df = read_csv(StringIO(_bytes.decode("utf-8")))
    except Exception as e:
        print(e)
        error = e
    return error, df


def highlight_bar(model_selected: Model, figure: Figure):
    '''The function highlights a bar in a figure if its x value matches the id of a selected model.
    
    Parameters
    ----------
    model_selected : Model
        The selected model object that needs to be highlighted in the figure.
    figure : Figure
        The `figure` parameter is likely an object that represents a plot or chart.
    Returns
    -------
        nothing
    
    '''
    if not model_selected:
        return
    for bar in figure["data"]:
        bar["alignmentgroup"] = "False"
        bar["marker"]["line"] = {}
        if bar["x"][0] == model_selected.id:
            bar["marker"]["line"]["color"] = "#f0bb65"
            bar["marker"]["line"]["width"] = 5


def new_style_inferences_table(ds: DataSession):
    '''This function creates a new style inferences table with conditional formatting based on the values
    of the prediction column and the y column in a given data session.
    
    '''
    return [
        {
            "if": {
                "filter_query": "{" + PRED_COLNAME + "} = {" + ds.y_col + "}",
                "column_id": PRED_COLNAME
            },
            "color": "#4eff00",
        },
        {
            "if": {
                "filter_query": "{" + PRED_COLNAME + "} > {" + ds.y_col + "} or {" + PRED_COLNAME + "} < {" + ds.y_col + "}",
                "column_id": PRED_COLNAME
            },
            "color": "red",
        }
    ]


def get_cool_palette(n_colors: int):
    palette = np.array(ORANGES)
    if n_colors > 5:
        return palette
    
    palette = np.array(np.split(palette, 2))

    return palette.T.ravel()