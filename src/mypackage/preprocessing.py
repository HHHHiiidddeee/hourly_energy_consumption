import numpy as np
import pandas as pd
import holidays

def get_missing_indices_diff(df : pd.DataFrame, column : str) -> np.ndarray:
    missing_indices = df[column][df[column].isna()].index.to_numpy()
    if len(missing_indices) == 0:
        return np.array([])
    diff = np.diff(missing_indices, prepend=-1)
    return diff

def get_continuous_missing_indices(df : pd.DataFrame, column : str) -> np.ndarray:
    diff = get_missing_indices_diff(df, column)
    continuous_missing = np.where(diff == 1)
    return continuous_missing

def get_continuous_missing_indices_all_columns(df : pd.DataFrame) -> dict:
    continuous_missing_dict = {}
    for column in df.columns:
        continuous_missing_dict[column] = get_continuous_missing_indices(df, column)
    return continuous_missing_dict

def get_start_index(df : pd.DataFrame, column : str) -> int:
    diff = get_missing_indices_diff(df, column)
    if len(np.where(diff != 1)[0]) == 0:
        return 0
    return np.where(diff != 1)[0][0]

def get_end_index(df : pd.DataFrame, column : str) -> int:
    diff = get_missing_indices_diff(df, column)
    if len(np.where(diff != 1)[0]) == 0:
        return len(df) - 1
    return np.where(diff != 1)[0][-1]

def get_effective_indices(df : pd.DataFrame, column : str) -> np.ndarray:
    non_missing_indices = df[column][df[column].notnull()].index.to_numpy()
    return np.arange(non_missing_indices[0], non_missing_indices[-1] + 1)

def get_effective_indices_all_columns(df : pd.DataFrame) -> dict:
    effective_indices_dict = {}
    for column in df.columns:
        effective_indices_dict[column] = get_effective_indices(df, column)
    return effective_indices_dict

def get_effective_series(df : pd.DataFrame, column : str) -> pd.Series:
    effective_indices = get_effective_indices(df, column)
    return df[column].iloc[effective_indices]

def get_effective_df(df : pd.DataFrame) -> pd.DataFrame:
    effective_indices_dict = get_effective_indices_all_columns(df)
    starts = []
    ends = []
    columns = []
    for column, indices in effective_indices_dict.items():
        if column == "PJM_Load" or column == "NI":
            continue
        columns.append(column)
        starts.append(indices[0])
        ends.append(indices[-1])
    return df.loc[max(starts):min(ends), columns].reset_index(drop=True)

def create_sin_cos_features(df: pd.DataFrame) -> pd.DataFrame:
    df = convert_datetime(df)
    df["day_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["day_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["week_sin"] = np.sin(2 * np.pi * (df["hour"] + df["dayofweek"] * 24) / (24 * 7))
    df["week_cos"] = np.cos(2 * np.pi * (df["hour"] + df["dayofweek"] * 24) / (24 * 7))
    df["month_sin"] = np.sin(2 * np.pi * (df["hour"] + (df["day"] - 1) * 24) / (24 * 30))
    df["month_cos"] = np.cos(2 * np.pi * (df["hour"] + (df["day"] - 1) * 24) / (24 * 30))
    df["half_year_sin"] = np.sin(2 * np.pi * (df["hour"] + (df["day"] - 1) * 24 + (df["month"] - 1) * 24 * 30) / (24 * 365 / 2))
    df["half_year_cos"] = np.cos(2 * np.pi * (df["hour"] + (df["day"] - 1) * 24 + (df["month"] - 1) * 24 * 30) / (24 * 365 / 2))
    df["year_sin"] = np.sin(2 * np.pi * (df["hour"] + (df["day"] - 1) * 24 + (df["month"] - 1) * 24 * 30) / (24 * 365))
    df["year_cos"] = np.cos(2 * np.pi * (df["hour"] + (df["day"] - 1) * 24 + (df["month"] - 1) * 24 * 30) / (24 * 365))
    return df

def convert_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if pd.api.types.is_datetime64_any_dtype(df["Datetime"]) and \
        df["Datetime"].is_monotonic_increasing:
        return df
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df

def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = convert_datetime(df)
    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek
    df["day"] = df["Datetime"].dt.day
    df["month"] = df["Datetime"].dt.month
    df["is_weekend"] = (df["Datetime"].dt.dayofweek >= 5).astype(int)
    df["is_holiday"] = df["Datetime"].dt.date.apply(lambda x: x in holidays.CountryHoliday("US")).astype(int)
    return df

def create_temporal_features(df: pd.DataFrame, lags) -> pd.DataFrame:
    df = convert_datetime(df)
    df["label"] = df["PJME_MW"].shift(-1)
    df = df.rename(columns = {"PJME_MW": "lag_1"})
    for lag in lags:
        df[f"lag_{lag}"] = df["lag_1"].shift(lag-1)
    df["diff_1h"] = df["lag_1"].diff()
    df = create_sin_cos_features(df)
    df["roll_mean_day"] = df["lag_1"].rolling(window=24).mean()
    df["roll_std_month"] = df["lag_1"].rolling(window=24*30).std()
    return df

def create_multioutput_labels(df: pd.DataFrame, pred_len: int):
    df = df.rename(columns={"label": "label_0"})
    subset = ["label_0"]
    for i in range(1, pred_len):
        df[f"label_{i}"] = df["label_0"].shift(-i)
        subset.append(f"label_{i}")
    return df.dropna(subset=subset).reset_index(drop=True)
