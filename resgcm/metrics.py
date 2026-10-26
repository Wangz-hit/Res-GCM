import numpy as np

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true-y_pred)**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true-y_pred)))
    denom = np.sum((y_true-np.mean(y_true))**2)
    r2 = float(1 - np.sum((y_true-y_pred)**2)/denom) if denom != 0 else float("nan")
    mape = float(np.mean(np.abs((y_true-y_pred)/y_true))*100)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}
