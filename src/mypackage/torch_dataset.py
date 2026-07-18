import numpy as np
import pandas as pd
from pyparsing import col
import torch
import torch.utils.data as Dataset
from .preprocessing import (
    convert_datetime,
    extract_datetime_features,
    create_sin_cos_features,
    create_temporal_features,
)

torch.manual_seed(42)
VALID_SIZE = 24 * 365
TEST_SIZE = 24 * 365

class EnergyDataset(Dataset.Dataset):
    def __init__(self, data: pd.DataFrame, seq_len: int, shift: int, pred_len: int, mode: str = "train"):
        self.seq_len = seq_len
        self.shift = shift
        self.pred_len = pred_len
        self.mode = mode
        
        if mode not in ["train", "test", "val", "re-train"]:
            raise ValueError("mode must be 'train', 'val', 'test', or 're-train'")
        
        self.preprocess(data)
        self._create_windows()
    
    def preprocess(self, data: pd.DataFrame):
        df = convert_datetime(data)
        df = extract_datetime_features(df)
        df = create_sin_cos_features(df)
        df = create_temporal_features(df, [2, 24, 24*7])
        df = df.dropna().reset_index(drop=True)
        df = df.drop(columns=["Datetime", "hour", "day", "dayofweek", "month"])
        
        self.retrain_data = df[:-TEST_SIZE].reset_index(drop=True)
        self.train_data, self.valid_data = self.retrain_data[:-VALID_SIZE].reset_index(drop=True), self.retrain_data[-VALID_SIZE:].reset_index(drop=True)
        self.test_data = df[-TEST_SIZE:].reset_index(drop=True)

        self._identify_binary_columns()

        self.normalize(self.retrain_data)
        self.normalize(self.train_data)
        self.normalize(self.valid_data)
        self.normalize(self.test_data)

        match self.mode:
            case "train":
                self.data = self.train_data.copy()
            case "val":
                self.data = self.valid_data.copy()
            case "test":
                self.data = self.test_data.copy()
            case "re-train":
                self.data = self.retrain_data.copy()

    def _identify_binary_columns(self):
        binary_cols = []
        continuous_cols = []
        
        for col in self.train_data.columns:
            unique_vals = self.train_data[col].nunique()
            if unique_vals <= 2 and set(self.train_data[col].unique()).issubset({0, 1}):
                binary_cols.append(col)
            else:
                continuous_cols.append(col)
        
        self.binary_cols = binary_cols
        self.continuous_cols = continuous_cols
        
        self.mean = {col: self.train_data[col].mean() for col in continuous_cols}
        self.std = {col: self.train_data[col].std() for col in continuous_cols}

    def _create_windows(self):
        self.windows = []

        if self.mode == "train":
            source_data = self.train_data.copy()
            target_len = len(self.train_data)
            start_offset = 0

        elif self.mode == "val":
            source_data = pd.concat([self.train_data, self.valid_data], ignore_index=True)
            target_len = len(self.valid_data)
            start_offset = len(self.train_data)

        elif self.mode == "test":
            source_data = pd.concat([self.retrain_data, self.test_data], ignore_index=True)
            target_len = len(self.test_data)
            start_offset = len(self.retrain_data)

        else:
            source_data = self.retrain_data.copy()
            target_len = len(self.retrain_data)
            start_offset = 0

        for i in range(target_len, -1, -self.shift):
            target_idx = start_offset + i - 1

            x_start = target_idx - self.pred_len + 1 - self.seq_len
            x_end = target_idx - self.pred_len + 1
            y_start = target_idx - self.pred_len
            y_end = target_idx

            if x_start < 0:
                continue

            x = source_data.drop(columns=["label"]).iloc[x_start:x_end].values
            y = source_data["label"].iloc[y_start:y_end].values

            if len(x) == self.seq_len and len(y) == self.pred_len:
                self.windows.append({
                    "x": torch.FloatTensor(x.copy()),
                    "y": torch.FloatTensor(y.copy()),
                })
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        return window['x'], window['y']
    
    def normalize(self, data: pd.DataFrame):
        for col in self.continuous_cols:
            data[col] = (data[col] - self.mean[col]) / self.std[col]
    
    def mode_switch(self, mode: str):
        if mode not in ["train", "test", "val", "re-train"]:
            raise ValueError("mode must be 'train', 'val', 'test', or 're-train'")
        self.mode = mode
        match self.mode:
            case "train":
                self.data = self.train_data.copy()
            case "val":
                self.data = self.valid_data.copy()
            case "test":
                self.data = self.test_data.copy()
            case "re-train":
                self.data = self.retrain_data.copy()
        self._create_windows()

    def get_retrain_data(self) -> pd.DataFrame:
        return self.retrain_data
    
    def get_train_data(self) -> pd.DataFrame:
        return self.train_data

    def get_valid_data(self) -> pd.DataFrame:
        return self.valid_data

    def get_test_data(self) -> pd.DataFrame:
        return self.test_data
    
    def inverse_transform(self, y: torch.Tensor) -> np.ndarray:
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        
        y_inv = y.copy()
        y_inv = y * self.std["label"] + self.mean["label"]
        
        return y_inv
