import pandas as pd


def get_unique_values(*args: pd.DataFrame) -> list:
    '''The function takes in multiple pandas dataframes as arguments, concatenates them, and returns a list
    of unique values for each column in the concatenated dataframe.
    
    Parameters
    ----------
     : pd.DataFrame
        The function `get_unique_values` takes in a variable number of arguments, each of which is expected
    to be a pandas DataFrame. The function concatenates all the DataFrames along the rows (axis=0) and
    then finds the unique values for each column in the resulting DataFrame. The function returns a
    
    Returns
    -------
        a list of unique values for each column in the concatenated dataframes passed as arguments.
    
    '''
    df = pd.concat(args, axis=0)
    uniques = [df[col].unique().tolist() for col in df.columns]
    return uniques
