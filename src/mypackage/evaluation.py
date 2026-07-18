import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, rmse, mape

def print_metrics(y_true, y_pred, title):
    mae, rmse, mape = calculate_metrics(y_true, y_pred)
    print(f"{title}:\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape*100:.2f}%\n")

def get_true_values(test_loader, dataset):
    y_trues = []
    for _, y in test_loader:
        y_trues.append(y.numpy())
    y_true = np.concatenate(y_trues, axis=0).reshape(-1)
    return dataset.inverse_transform(y_true)
