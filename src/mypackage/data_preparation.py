import numpy as np
import pandas as pd
from typing import TypedDict, Union
from .preprocessing import (
    convert_datetime,
    extract_datetime_features,
    create_sin_cos_features,
    create_temporal_features,
    create_multioutput_labels
)

TEST_SIZE = 24 * 365

class LGBMData(TypedDict):
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray

def prepare_statistical_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = convert_datetime(df)
    train = df[:-TEST_SIZE]
    test = df[-TEST_SIZE:]
    return train, test

def prepare_prophet_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = convert_datetime(df)
    df = extract_datetime_features(df)
    df = df.rename(columns={"Datetime": "ds", "PJME_MW": "y"})
    train = df[:-TEST_SIZE]
    test = df[-TEST_SIZE:]
    return train, test

def prepare_lightgbm_data(df: pd.DataFrame, lags: Union[list, tuple, np.ndarray], pred_len: int, shift: int = 1) -> LGBMData:
    df = convert_datetime(df)
    df = extract_datetime_features(df)
    df = create_sin_cos_features(df)
    df = create_temporal_features(df, lags)
    df = df.drop(columns=["hour", "day", "dayofweek", "month"])
    df = create_multioutput_labels(df, pred_len)
    train = df[:-TEST_SIZE].reset_index(drop=True)
    train = train[::shift]
    test = df[-TEST_SIZE:].reset_index(drop=True)
    test = test[::pred_len]
    labels = [f"label_{i}" for i in range(pred_len)]
    data = {"X_train": train.drop(columns=labels), "y_train": train[labels].values,
            "X_test": test.drop(columns=labels), "y_test": test[labels].values}
    return data
