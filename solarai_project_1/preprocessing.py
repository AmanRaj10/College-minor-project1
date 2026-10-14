import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
import os

# ── Path where feature list is saved alongside the model ──────────────────────
FEATURE_NAMES_PATH = "saved_model/feature_names.json"


def engineer_features(df, lat=20.0):
    """
    Deterministic feature engineering — always produces the same columns
    regardless of which optional columns (Temperature, Humidity, etc.) are
    present in the raw DataFrame.  Missing optional columns are filled with
    sensible defaults so the output shape is always identical.

    Features (always present, always in this order):
      Core raw     : Solar_Output, GHI, Clear_Sky_GHI, Temperature, Humidity,
                     WindSpeed, Pressure, Cloud_Cover
      Time scalars : hour, day_of_year, month, day_of_week, week_of_year,
                     is_weekend, quarter
      Cyclical     : hour_sin/cos, month_sin/cos, doy_sin/cos, dow_sin/cos
      Solar geom   : solar_declination, hour_angle, solar_elevation
      Derived      : cloud_cover_index, Solar_Output (ensured)
      Interactions : temp_ghi_interaction, temp_normalized,
                     humidity_ghi, wind_ghi
      Lags GHI     : 1,2,3,6,12,24,48 h
      Lags Solar   : 1,24 h
      Rolling GHI  : mean/std/max for 3,6,12,24 h windows
      Rolling Sol  : mean 24 h
      Diffs        : GHI_diff_1h, GHI_diff_24h
      Flags        : is_daytime, is_peak_solar
    """
    df = df.copy()

    # ── Datetime index ───────────────────────────────────────────────────────
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range(start='2022-01-01', periods=len(df), freq='h')

    # ── Ensure all raw columns exist (fill missing with defaults) ────────────
    if 'GHI' not in df.columns:
        df['GHI'] = 0.0
    df['GHI'] = df['GHI'].clip(lower=0)

    if 'Clear_Sky_GHI' not in df.columns:
        df['Clear_Sky_GHI'] = df['GHI'] * 1.15
    df['Clear_Sky_GHI'] = df['Clear_Sky_GHI'].clip(lower=0)

    if 'Solar_Output' not in df.columns:
        df['Solar_Output'] = df['GHI'] * 0.20
    df['Solar_Output'] = df['Solar_Output'].clip(lower=0)

    # Optional meteorological columns — fill with neutral defaults if absent
    if 'Temperature' not in df.columns:
        df['Temperature'] = 25.0
    if 'Humidity' not in df.columns:
        df['Humidity'] = 50.0
    if 'WindSpeed' not in df.columns:
        df['WindSpeed'] = 3.0
    if 'Pressure' not in df.columns:
        df['Pressure'] = 101.3
    if 'Cloud_Cover' not in df.columns:
        df['Cloud_Cover'] = np.where(
            df['Clear_Sky_GHI'] > 5,
            np.clip(1 - df['GHI'] / (df['Clear_Sky_GHI'] + 1e-6), 0, 1),
            0.0
        )

    # ── Time scalars ─────────────────────────────────────────────────────────
    df['hour']         = df.index.hour
    df['day_of_year']  = df.index.dayofyear
    df['month']        = df.index.month
    df['day_of_week']  = df.index.dayofweek
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
    df['quarter']      = df.index.quarter

    # ── Cyclical encoding ─────────────────────────────────────────────────────
    df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin']   = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos']   = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['dow_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # ── Solar geometry ────────────────────────────────────────────────────────
    B = 360 / 365 * (df['day_of_year'] - 81)
    df['solar_declination'] = 23.45 * np.sin(np.deg2rad(B))
    df['hour_angle']        = 15 * (df['hour'] - 12)
    lat_rad = np.deg2rad(lat)
    dec_rad = np.deg2rad(df['solar_declination'])
    ha_rad  = np.deg2rad(df['hour_angle'])
    df['solar_elevation'] = np.clip(
        np.sin(lat_rad) * np.sin(dec_rad) +
        np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad),
        0, 1
    )

    # ── Cloud cover index ─────────────────────────────────────────────────────
    df['cloud_cover_index'] = np.where(
        df['Clear_Sky_GHI'] > 5,
        np.clip(1 - df['GHI'] / (df['Clear_Sky_GHI'] + 1e-6), 0, 1),
        0.0
    )

    # ── Interaction features (always computed — cols always present now) ──────
    ghi_max = df['GHI'].max()
    safe_max = ghi_max if ghi_max > 0 else 1.0
    df['temp_ghi_interaction'] = df['Temperature'] * df['GHI'] / safe_max
    temp_std = df['Temperature'].std()
    df['temp_normalized'] = (
        (df['Temperature'] - df['Temperature'].mean()) / (temp_std if temp_std > 0 else 1.0)
    )
    df['humidity_ghi'] = df['Humidity'] * df['GHI'] / safe_max
    df['wind_ghi']     = df['WindSpeed'] * df['GHI'] / safe_max

    # ── Lag features ─────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f'GHI_lag_{lag}h'] = df['GHI'].shift(lag).fillna(0)
    for lag in [1, 24]:
        df[f'Solar_lag_{lag}h'] = df['Solar_Output'].shift(lag).fillna(0)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for w in [3, 6, 12, 24]:
        roll = df['GHI'].rolling(w, min_periods=1)
        df[f'GHI_roll_{w}h'] = roll.mean()
        df[f'GHI_std_{w}h']  = roll.std().fillna(0)
        df[f'GHI_max_{w}h']  = roll.max()
    df['Solar_roll_24h'] = df['Solar_Output'].rolling(24, min_periods=1).mean()

    # ── Difference features ───────────────────────────────────────────────────
    df['GHI_diff_1h']  = df['GHI'].diff(1).fillna(0)
    df['GHI_diff_24h'] = df['GHI'].diff(24).fillna(0)

    # ── Binary flags ─────────────────────────────────────────────────────────
    df['is_daytime']   = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
    df['is_peak_solar'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(int)

    df = df.fillna(0)
    return df


def _align_features(df_numeric, saved_feature_names):
    """
    Align df_numeric columns to exactly match saved_feature_names:
      - Add any missing columns as zeros
      - Drop any extra columns
      - Reorder to match training order exactly
    This guarantees shape (N, len(saved_feature_names)) regardless of data source.
    """
    for col in saved_feature_names:
        if col not in df_numeric.columns:
            df_numeric[col] = 0.0
    df_numeric = df_numeric[saved_feature_names]
    return df_numeric


def prepare_sequences(df, window=24, target_col='Solar_Output',
                       scaler_path=None, feature_names_path=None):
    """
    Prepare windowed sequences for training.
    Saves feature names to JSON so inference can reproduce the exact same shape.
    """
    df_feat    = engineer_features(df)
    df_numeric = df_feat.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])

    # Target column first
    if target_col in df_numeric.columns:
        cols = [target_col] + [c for c in df_numeric.columns if c != target_col]
        df_numeric = df_numeric[cols]

    feature_names = df_numeric.columns.tolist()
    print(f"Feature count: {len(feature_names)}")
    print(f"Features: {feature_names}")

    # ── Save feature list so inference can align to it ────────────────────────
    fnames_path = feature_names_path or FEATURE_NAMES_PATH
    os.makedirs(os.path.dirname(fnames_path), exist_ok=True)
    with open(fnames_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved → {fnames_path}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df_numeric.values)

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i])
        y.append(scaled[i, 0])

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            scaler,
            feature_names)


def get_inference_features(df, scaler, feature_names_path=None):
    """
    Prepare features for inference (forecast tab).
    Loads the saved feature list and aligns the current DataFrame to match it
    exactly — guaranteeing the same shape as during training, even if the
    incoming data has different optional columns.
    """
    fnames_path = feature_names_path or FEATURE_NAMES_PATH

    if not os.path.exists(fnames_path):
        raise FileNotFoundError(
            f"Feature names file not found: {fnames_path}\n"
            "Please retrain the model with `python train.py` first."
        )

    with open(fnames_path) as f:
        saved_feature_names = json.load(f)

    df_feat    = engineer_features(df)
    df_numeric = df_feat.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])

    # Align to exact training feature set
    df_numeric = _align_features(df_numeric, saved_feature_names)

    scaled = scaler.transform(df_numeric.values)
    return scaled, saved_feature_names


def load_multi_year_data(paths):
    """Load and concatenate multiple year CSV files."""
    dfs = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"  Loaded: {path} ({len(df):,} rows)")
        except Exception as e:
            print(f"  ⚠️ Failed: {path} — {e}")
    if not dfs:
        raise ValueError("No valid CSV files found!")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined dataset: {len(combined):,} rows")
    return combined


def battery_optimization(predictions_kw, battery_capacity_kwh=10.0, max_charge_rate=5.0,
                          grid_buy_price=8.0, grid_sell_price=3.5):
    """
    Rule-based battery optimization strategy.
    
    Strategy:
    - Charge when solar > 120% of average (excess solar available)
    - Discharge when solar < 50% of average AND during peak hours
    - Track SOC with 10–90% bounds for battery health
    
    Returns DataFrame with hourly schedule and savings.
    """
    schedule = []
    battery_soc = battery_capacity_kwh * 0.5  # Start at 50%
    avg_pred = np.mean(predictions_kw[predictions_kw > 0]) if np.any(predictions_kw > 0) else np.mean(predictions_kw)
    
    for i, power in enumerate(predictions_kw):
        hour_of_day = i % 24
        is_peak = 6 <= hour_of_day <= 22
        is_high_tariff = (7 <= hour_of_day <= 11) or (18 <= hour_of_day <= 22)

        soc_pct = battery_soc / battery_capacity_kwh
        can_charge = soc_pct < 0.90
        can_discharge = soc_pct > 0.10

        if power > avg_pred * 1.2 and can_charge:
            # Excess solar → charge battery
            charge = min(power - avg_pred, max_charge_rate, battery_capacity_kwh * 0.90 - battery_soc)
            charge = max(0, charge)
            battery_soc += charge
            action = f"Charge +{charge:.2f} kWh"
            net_to_grid = max(0, power - charge)
            savings = charge * grid_buy_price  # Avoided grid purchase

        elif power < avg_pred * 0.5 and is_peak and can_discharge:
            # Low solar during peak → discharge battery
            need = avg_pred - power
            discharge = min(need, max_charge_rate, battery_soc - battery_capacity_kwh * 0.10)
            discharge = max(0, discharge)
            battery_soc -= discharge
            action = f"Discharge -{discharge:.2f} kWh"
            net_to_grid = 0
            savings = discharge * (grid_buy_price if is_high_tariff else grid_sell_price)

        else:
            action = "Idle"
            net_to_grid = power
            savings = 0.0

        schedule.append({
            'hour': i,
            'hour_of_day': hour_of_day,
            'solar_kw': round(float(power), 3),
            'battery_soc_kwh': round(float(battery_soc), 3),
            'battery_soc_pct': round(float(battery_soc / battery_capacity_kwh * 100), 1),
            'action': action,
            'net_to_grid_kw': round(float(net_to_grid), 3),
            'estimated_savings_inr': round(float(savings), 2),
            'is_peak_hour': is_peak,
            'is_high_tariff': is_high_tariff
        })

    return pd.DataFrame(schedule)