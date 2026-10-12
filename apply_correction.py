"""
Apply Gamma Quantile Mapping Correction to Hourly ERA5-Land Timeseries
=======================================================================

Input  : hourly ERA5-Land precipitation (Excel or CSV)
Output : hourly bias-corrected precipitation

Workflow:
    1. Load fitted Gamma QM parameters from gamma_qm_params.json
         (produced by gamma_quantile_mapping_auto.py)
    2. Load hourly ERA5-Land timeseries
    3. Aggregate hourly → daily accumulated precipitation
    4. Compute hourly factor  =  hourly_precip / daily_total
         (fraction of the day's total that fell in each hour)
    5. Apply Gamma QM on the daily totals
    6. Corrected hourly  =  corrected_daily_total × hourly_factor
    7. Save corrected hourly output

Usage:
    1. Run gamma_quantile_mapping_auto.py first to generate gamma_qm_params.json
    2. Set INPUT_FILE, DATETIME_COL, and ERA5_COL below
    3. Run: python apply_correction.py
"""

import numpy as np
import pandas as pd
import os
import json
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
# JSON file produced by gamma_quantile_mapping_auto.py
PARAMS_FILE = r"D:\Masters\data\predicting_flashflood_events_codes\gamma_qm_params.json"

# New ERA5-Land timeseries to correct (Excel or CSV)
INPUT_FILE  = r"D:\Masters\data\rainfall\sari-era5.xlsx"

# Column containing the datetime (hourly timestamps)
DATETIME_COL = "Date"          # change to match your column name

# Column containing hourly ERA5-Land precipitation
ERA5_COL     = "precip"        # change if your column is named differently
LEGACY_COL   = "tp_daily"      # fallback if ERA5_COL not found

# Daily accumulation window end hour (local time)
# Example with 6: 2010-01-02 total uses [2010-01-01 06:00, 2010-01-02 06:00)
ACCUM_END_HOUR = 6

# Output corrected column name
OUT_COL      = "corr_precip"

# Output file (same directory as PARAMS_FILE by default)
OUTPUT_FILE = os.path.join(
    os.path.dirname(PARAMS_FILE),
    "era5_corrected_hourly.csv"
)
# ─────────────────────────────────────────────────────────────────────────────


def load_params(params_file):
    """Load fitted gamma QM parameters from JSON."""
    if not os.path.isfile(params_file):
        raise FileNotFoundError(
            f"Parameters file not found: {params_file}\n"
            "Run gamma_quantile_mapping_auto.py first to generate it."
        )
    with open(params_file) as f:
        p = json.load(f)

    required = ["obs_threshold", "sim_threshold",
                "obs_gamma_shape", "obs_gamma_loc", "obs_gamma_scale",
                "obs_wet_fraction"]
    for key in required:
        if key not in p:
            raise KeyError(f"Expected key '{key}' missing from {params_file}")

    print("Loaded parameters:")
    print(f"  obs_threshold    : {p['obs_threshold']:.4f} mm")
    print(f"  sim_threshold    : {p['sim_threshold']:.4f} mm")
    print(f"  obs_gamma_shape  : {p['obs_gamma_shape']:.4f}")
    print(f"  obs_gamma_scale  : {p['obs_gamma_scale']:.4f}")
    print(f"  obs_wet_fraction : {p['obs_wet_fraction']*100:.2f}%")
    return p


def load_timeseries(filepath, datetime_col, era5_col, legacy_col):
    """Load hourly Excel or CSV timeseries and return a DataFrame with parsed datetime."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".xls", ".xlsx"):
        df = pd.read_excel(filepath)
    elif ext == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .xlsx or .csv")

    # Parse datetime column
    if datetime_col not in df.columns:
        raise KeyError(
            f"Datetime column '{datetime_col}' not found.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Update DATETIME_COL in the configuration section."
        )
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    n_bad_dates = df[datetime_col].isna().sum()
    if n_bad_dates > 0:
        print(f"  WARNING: {n_bad_dates} rows have unparseable timestamps and will be dropped.")
        df = df[df[datetime_col].notna()].copy()

    if era5_col not in df.columns:
        if legacy_col in df.columns:
            print(f"  NOTE: '{era5_col}' not found. Using fallback column '{legacy_col}'.")
            df[era5_col] = df[legacy_col]
        else:
            available = list(df.columns)
            raise KeyError(
                f"Column '{era5_col}' not found in {filepath}.\n"
                f"Available columns: {available}\n"
                f"Update ERA5_COL in the configuration section."
            )

    df[era5_col] = pd.to_numeric(df[era5_col], errors="coerce").fillna(0)
    return df


def compute_hourly_factors(df, datetime_col, era5_col, accum_end_hour=ACCUM_END_HOUR):
    """
    Aggregate hourly precipitation to daily totals and compute hourly factors.

    hourly_factor = hourly_precip / daily_total
    If the daily total is 0 (dry day), hourly_factor = 0 for all hours of that day.

    Parameters
    ----------
    df           : DataFrame with parsed datetime and ERA5 precip columns
    datetime_col : name of the datetime column
    era5_col     : name of the hourly precipitation column

    Returns
    -------
    daily_df     : DataFrame indexed by date with daily accumulated totals (column era5_col)
    df           : original hourly DataFrame with added 'hourly_factor' and '_date' columns
    """
    df = df.copy()

    # Hydrological-day label with custom end hour.
    # For accum_end_hour=6, accumulation day D is [D-1 06:00, D 06:00).
    shifted = df[datetime_col] - pd.to_timedelta(accum_end_hour, unit="h")
    df["_date"] = (shifted.dt.floor("D") + pd.Timedelta(days=1)).dt.date

    # Daily accumulated totals
    daily_total = df.groupby("_date")[era5_col].sum()
    df["_daily_total"] = df["_date"].map(daily_total)

    # Hourly fraction: how much of the day's rain fell in this hour
    df["hourly_factor"] = np.where(
        df["_daily_total"] > 0,
        df[era5_col] / df["_daily_total"],
        0.0
    )

    # Build daily DataFrame for gamma QM
    daily_df = daily_total.reset_index()
    daily_df.columns = ["_date", era5_col]

    return daily_df, df


def apply_gamma_qm_daily(daily_series, params):
    """
    Apply gamma quantile mapping to a daily accumulated precipitation Series.

    Algorithm:
      1. Filter ERA5-Land wet days (> sim_threshold)
      2. Fit gamma to those wet days
      3. Map each wet day quantile: ERA5-Land gamma → observed gamma
      4. Dry days remain zero

    Parameters
    ----------
    daily_series : pd.Series  Daily accumulated ERA5-Land precipitation.
    params       : dict       Fitted parameters from gamma_qm_params.json.

    Returns
    -------
    np.ndarray — bias-corrected daily values (same length as daily_series)
    """
    sim_threshold = params["sim_threshold"]
    obs_shape     = params["obs_gamma_shape"]
    obs_loc       = params["obs_gamma_loc"]
    obs_scale     = params["obs_gamma_scale"]

    sim_vals  = daily_series.values
    corrected = np.zeros(len(sim_vals))
    all_idx   = np.arange(len(sim_vals))

    keep_wet = all_idx[sim_vals > sim_threshold]
    n_kept   = len(keep_wet)

    print(f"\n  Total days      : {len(sim_vals)}")
    print(f"  Wet days (>{sim_threshold:.3f} mm): {n_kept}  ({n_kept/len(sim_vals)*100:.1f}%)")
    print(f"  Dry days        : {len(sim_vals) - n_kept}")

    if n_kept < 10:
        raise ValueError(
            f"Only {n_kept} wet days found (threshold={sim_threshold:.3f} mm). "
            "Too few to fit the ERA5-Land distribution. "
            "Check that INPUT_FILE contains ERA5-Land data for a comparable region/period."
        )

    subset = sim_vals[keep_wet]
    sub_shape, sub_loc, sub_scale = stats.gamma.fit(subset, floc=0)

    print(f"\n  ERA5-Land gamma (this period):")
    print(f"    shape = {sub_shape:.4f},  scale = {sub_scale:.4f}")
    print(f"  Target (observed) gamma:")
    print(f"    shape = {obs_shape:.4f},  scale = {obs_scale:.4f}")

    for idx in keep_wet:
        q = stats.gamma.cdf(sim_vals[idx], sub_shape, sub_loc, sub_scale)
        q = np.clip(q, 1e-6, 1 - 1e-6)
        corrected[idx] = stats.gamma.ppf(q, obs_shape, obs_loc, obs_scale)

    n_bad = np.sum(np.isnan(corrected) | np.isinf(corrected))
    if n_bad:
        print(f"\n  WARNING: {n_bad} invalid values set to 0.")
        corrected = np.where(np.isfinite(corrected), corrected, 0.0)

    return corrected


def redistribute_to_hourly(hourly_df, daily_df, corrected_daily, era5_col):
    """
    Multiply each corrected daily total by the hourly factor to get
    corrected hourly precipitation.

    corrected_hourly = corrected_daily_total × hourly_factor
    """
    date_to_corrected = dict(zip(daily_df["_date"], corrected_daily))
    hourly_df = hourly_df.copy()
    hourly_df["_corr_daily"] = hourly_df["_date"].map(date_to_corrected).fillna(0.0)
    hourly_df[OUT_COL] = hourly_df["_corr_daily"] * hourly_df["hourly_factor"]

    # Keep accumulated daily columns in final output for traceability
    hourly_df.rename(
        columns={
            "_date": "accum_date",
            "_daily_total": "daily_accum_raw",
            "_corr_daily": "daily_accum_corrected",
        },
        inplace=True,
    )
    return hourly_df


def print_summary(hourly_df, daily_df, corrected_daily, era5_col, params):
    """Print before/after statistics at both daily and hourly level."""
    obs_thr = params["obs_threshold"]
    sim_thr = params["sim_threshold"]

    print("\n" + "="*60)
    print("BEFORE / AFTER SUMMARY")
    print("="*60)

    raw_daily = daily_df[era5_col]
    sim_wet   = raw_daily.values[raw_daily.values > sim_thr]
    corr_wet  = corrected_daily[corrected_daily > obs_thr]

    print(f"\n  Daily ERA5-Land (raw):   mean={raw_daily.mean():.3f} mm/day  "
          f"max={raw_daily.max():.1f} mm")
    if len(sim_wet):
        print(f"    wet-day mean      : {sim_wet.mean():.3f} mm")
        print(f"    wet-day 95th pct  : {np.percentile(sim_wet, 95):.3f} mm")

    print(f"\n  Daily corrected:         mean={corrected_daily.mean():.3f} mm/day  "
          f"max={corrected_daily.max():.1f} mm")
    if len(corr_wet):
        print(f"    wet-day mean      : {corr_wet.mean():.3f} mm")
        print(f"    wet-day 95th pct  : {np.percentile(corr_wet, 95):.3f} mm")

    raw_hourly  = hourly_df[era5_col]
    corr_hourly = hourly_df[OUT_COL]
    print(f"\n  Hourly ERA5-Land (raw):  mean={raw_hourly.mean():.4f} mm/hr  "
          f"max={raw_hourly.max():.3f} mm/hr")
    print(f"  Hourly corrected:        mean={corr_hourly.mean():.4f} mm/hr  "
          f"max={corr_hourly.max():.3f} mm/hr")

    print(f"\n  Wet-day frequency:")
    print(f"    ERA5-Land raw : {(raw_daily > sim_thr).mean()*100:.2f}%")
    print(f"    Corrected     : {(corrected_daily > obs_thr).mean()*100:.2f}%")
    print(f"    Target        : {params['obs_wet_fraction']*100:.2f}%")
    print("="*60)


def main():
    print("\n" + "="*60)
    print("GAMMA QM — APPLY CORRECTION TO NEW TIMESERIES")
    print("="*60)

    # 1. Load parameters
    print("\n[1] Loading fitted parameters...")
    params = load_params(PARAMS_FILE)

    # 2. Load new timeseries
    print(f"\n[2] Loading timeseries from: {INPUT_FILE}")
    df = load_timeseries(INPUT_FILE, DATETIME_COL, ERA5_COL, LEGACY_COL)
    print(f"    {len(df)} hourly rows loaded   |   columns: {list(df.columns)}")

    # 3. Aggregate to daily and compute hourly factors
    print("\n[3] Aggregating to daily totals and computing hourly factors...")
    daily_df, df = compute_hourly_factors(df, DATETIME_COL, ERA5_COL, ACCUM_END_HOUR)
    print(f"    {len(daily_df)} days  |  {len(df)} hourly rows")
    print(f"    Daily accumulation boundary: {ACCUM_END_HOUR:02d}:00")

    # 4. Apply gamma QM on daily totals
    print("\n[4] Applying gamma QM on daily accumulated totals...")
    corrected_daily = apply_gamma_qm_daily(daily_df[ERA5_COL], params)

    # 5. Redistribute corrected daily totals back to hourly
    print("\n[5] Redistributing corrected totals back to hourly...")
    df = redistribute_to_hourly(df, daily_df, corrected_daily, ERA5_COL)

    # 6. Summary
    print_summary(df, daily_df, corrected_daily, ERA5_COL, params)

    # 7. Save
    ext = os.path.splitext(OUTPUT_FILE)[1].lower()
    if ext == ".csv":
        df.to_csv(OUTPUT_FILE, index=False)
    else:
        df.to_excel(OUTPUT_FILE, index=False)

    print(f"\n[6] Output saved → {OUTPUT_FILE}")
    print("="*60)


if __name__ == "__main__":
    main()
