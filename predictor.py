import pandas as pd
import numpy as np
from xgboost import XGBRegressor


def run_forecast(df: pd.DataFrame, target_column: str, forecast_months: int) -> pd.DataFrame:

    # ── Build date index ─────────────────────────────────────────
    df = df.copy()
    df["Date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str),
        errors="coerce"
    )
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 3:
        raise ValueError(
            f"Not enough data rows to train a forecast model. "
            f"Got {len(df)} rows — need at least 3."
        )

    # ── Feature engineering ──────────────────────────────────────
    df["Month_Number"]   = df["Date"].dt.month
    df["Time_Index"]     = np.arange(len(df))
    df["Lag1"]           = df[target_column].shift(1)
    df["Lag2"]           = df[target_column].shift(2)
    df["Growth"]         = df[target_column].pct_change()
    df["Rolling_Mean_3"] = df[target_column].rolling(3).mean()
    df["Rolling_Std_3"]  = df[target_column].rolling(3).std()
    df["Month"] = df["Month"].fillna("December")
    # Use forward-fill then back-fill (replaces deprecated fillna(method=...))
    df = df.ffill().bfill()

    features = [
        "Time_Index",
        "Month_Number",
        "Lag1",
        "Lag2",
        "Growth",
        "Rolling_Mean_3",
        "Rolling_Std_3",
    ]

    X = df[features]
    y = df[target_column]

    # ── Train model ──────────────────────────────────────────────
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        verbosity=0,
    )
    model.fit(X, y)

    # ── Iterative forecast ───────────────────────────────────────
    last_row  = df.iloc[-1].copy()
    last_date = last_row["Date"]

    predictions  = []
    future_dates = []

    for _ in range(forecast_months):

        next_date = last_date + pd.DateOffset(months=1)

        new_row = last_row.copy()
        new_row["Date"]         = next_date
        new_row["Month_Number"] = next_date.month

        X_future = pd.DataFrame(
            [new_row[features].values],
            columns=features
        )
        pred = float(model.predict(X_future)[0])

        predictions.append(pred)
        future_dates.append(next_date)

        # Update lag / rolling features for next step
        lag1 = float(new_row["Lag1"])
        lag2 = float(new_row["Lag2"])

        new_row["Lag2"]           = lag1
        new_row["Lag1"]           = pred
        new_row["Growth"]         = (pred - lag1) / abs(lag1) if lag1 != 0 else 0.0
        new_row["Rolling_Mean_3"] = (pred + lag1 + lag2) / 3
        new_row["Rolling_Std_3"]  = float(np.std([pred, lag1, lag2]))
        new_row["Time_Index"]     = last_row["Time_Index"] + 1

        last_row  = new_row
        last_date = next_date

    # ── Return forecast DataFrame ────────────────────────────────
    forecast_df = pd.DataFrame({
        "Date":                       future_dates,
        f"Predicted {target_column}": predictions,
    })

    return forecast_df