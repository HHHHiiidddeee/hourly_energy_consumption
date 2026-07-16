import torch
import numpy as np
import pandas as pd

def dnn_forecast(model, test_loader, device):
    model.eval()
    y_preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            y_pred = model(x)
            y_preds.append(y_pred.cpu().numpy())
    y_pred = np.concatenate(y_preds, axis=0).reshape(-1)
    return test_loader.inverse_transform(y_pred)

def lightgbm_forecast(model, data, pred_len):
    idx = len(data["X_train"])
    predict_df = pd.concat([data["X_train"], data["X_test"]], axis=0, ignore_index=True)
    q, r = divmod(len(data["X_test"]), pred_len)
    y_preds = []
    if r != 0:
        q += 1
    for i in range(q):
        if i == q - 1 and r != 0:
            pred_len = r
        for j in range(pred_len):
            y_pred = model.predict(predict_df.iloc[[idx], 1:])
            y_preds.append(y_pred)
            if i == q-1 and j == pred_len - 1:
                break
            idx += 1
            fill_predict_df_by_pred(predict_df, y_pred, idx, lags=[2, 24, 24*7])
        fill_predict_df_by_trues(predict_df, data["y_test"][i * pred_len:i * pred_len + pred_len].reshape(-1), 
                                 idx, lags=[2, 24, 24*7], pred_len=pred_len)
    return np.array(y_preds).reshape(-1)

def fill_predict_df_by_pred(predict_df, y_pred, idx, lags):
    predict_df.loc[predict_df.index[idx], "lag_1"] = y_pred

    for lag in lags:
        predict_df.loc[predict_df.index[idx], f"lag_{lag}"] = predict_df.loc[predict_df.index[idx-lag], "lag_1"]

    predict_df.loc[predict_df.index[idx], "diff_1h"] = (
        predict_df.iloc[idx]["lag_1"]
        - predict_df.iloc[idx-1]["lag_1"]
    )
    predict_df.loc[predict_df.index[idx], "roll_mean_day"] = (
        predict_df.loc[
            predict_df.index[idx-23:idx+1],
            "lag_1"
        ].mean()
    )
    predict_df.loc[predict_df.index[idx], "roll_std_month"] = (
        predict_df.loc[
            predict_df.index[idx-24*30+1:idx+1],
            "lag_1"
        ].std()
    )

def fill_predict_df_by_trues(predict_df, y_trues, idx, lags, pred_len):
    start = idx - pred_len + 1
    end = idx
    rows = predict_df.index[start:end+1]

    predict_df.loc[rows, "lag_1"] = y_trues

    for lag in lags:
        predict_df.loc[rows, f"lag_{lag}"] = predict_df.loc[predict_df.index[start-lag:end-lag+1], "lag_1"].values

    predict_df.loc[rows, "diff_1h"] = (
        predict_df.loc[rows, "lag_1"].values
        - predict_df.loc[
            predict_df.index[start-1:end],
            "lag_1"
        ].values
    )

    for i in range(start, end+1):
        predict_df.loc[predict_df.index[i], "roll_mean_day"] = (
            predict_df.loc[
                predict_df.index[i-23:i+1],
                "lag_1"
            ].mean()
        )
        predict_df.loc[predict_df.index[i], "roll_std_month"] = (
            predict_df.loc[
                predict_df.index[i-24*30+1:i+1],
                "lag_1"
            ].std()
        )

def sarimax_forecast(result, test_data, pred_len):
    y_preds = []
    start = 0
    q, r = divmod(len(test_data), pred_len)
    if r != 0:
        q += 1
    for i in range(q):
        if i == q - 1 and r != 0:
            pred_len = r
        end = start + pred_len
        y_pred = result.forecast(steps=pred_len)
        y_preds.extend(y_pred)
        result = result.extend(test_data["PJME_MW"][start:end])
        start = end
    return np.array(y_preds)

def baseline_forecast(y_train, y_test, len_pred):
    q, r = divmod(len(y_test), len_pred)
    whole = np.concatenate([y_train, y_test], axis=0)
    if r > 0:
        q += 1
    else:
        r = len_pred
    pred = []
    for t in range(q):
        start = len(y_train) - len_pred + t * len_pred
        if t == q - 1:
            end = start + r
        else:
            end = start + len_pred
        pred.extend(whole[start:end])
    return pred
