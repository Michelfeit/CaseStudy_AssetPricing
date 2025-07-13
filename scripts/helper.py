import numpy as np
import pandas as pd


def get_elements_of_series(series:pd.Series, elements:np.array):
    r = []
    for e in elements:
        r.append(series[series == e])
    return r

def delete_elements_from_series(series:pd.Series, elements:np.array, inplace:bool=True):
    r = series
    if not inplace:
        r = series.copy(deep=True)
    for e in elements:
        r.drop(e.index, inplace=True)
    return r

def delete_columns_from_dataframe(df:pd.DataFrame, elements:np.array, inplace:bool=True):
    r = df
    if not inplace:
        r = df.copy(deep=True)
    for e in elements:
        r.drop(e.index[0], axis=1, inplace=True)
    return r

