from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "temp_diff",
    "wear_rate",
    "torque_temp_interaction",
    "torque_roll_mean_5",
    "torque_lag_1",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["temp_diff"] = x["process_temperature"] - x["air_temperature"]
    x["wear_rate"] = x["tool_wear"] / (x["rotational_speed"].abs() + 1.0)
    x["torque_temp_interaction"] = x["torque"] * x["process_temperature"]
    x["torque_roll_mean_5"] = x["torque"].rolling(window=5, min_periods=1).mean()
    x["torque_lag_1"] = x["torque"].shift(1).bfill()
    return x
