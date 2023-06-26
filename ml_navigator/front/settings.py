import os
import pandas as pd

bag_of_model_names = pd.read_csv(os.path.join(os.getcwd(), "ml_navigator", "front", "assets", "data", "stars.csv")).sample(frac=1)["Stars"].str.lower().values.tolist()

PRED_COLNAME = "y pred"