"""
Gamma Quantile Mapping Bias Correction with Automatic Wet-Day Threshold
=========================================================================

This script automatically determines the wet-day threshold from observed data
instead of using an arbitrary value. The threshold is set such that the 
corrected data matches the observed wet-day frequency.

Key improvements:
1. Automatic wet-day threshold computation
2. Wet-day frequency matching
3. Proper gamma distribution fitting
4. Comprehensive validation metrics
"""

import numpy as np
import pandas as pd
import os
import json
from scipy import stats
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────
DATA_FILE  = r"D:\Masters\data\rainfall\merged_obs_df_era5_on_dates.xlsx"
OBS_COL    = "rainfall"       # observed rainfall column
SIM_COL    = "tp_rainfall"    # ERA5-Land rainfall column
LEGACY_SIM_COL = "tp_daily"   # fallback for older files
OUT_COL    = "corr_tp_daily"  # output corrected column
MIN_THRESHOLD = 0.1           # minimum physical threshold (mm) - instrument detection limit
# ─────────────────────────────────────────────────────────────────────────────


def load_data(filepath):
    """Load Excel file and return a clean DataFrame."""
    df = pd.read_excel(filepath)

    if SIM_COL not in df.columns and LEGACY_SIM_COL in df.columns:
        print(f"  NOTE: '{SIM_COL}' not found. Using fallback column '{LEGACY_SIM_COL}'.")
        df[SIM_COL] = df[LEGACY_SIM_COL]

    df[OBS_COL] = pd.to_numeric(df[OBS_COL], errors="coerce").fillna(0)
    df[SIM_COL] = pd.to_numeric(df[SIM_COL], errors="coerce").fillna(0)
    return df


def compute_wet_day_threshold(obs_series, sim_series, min_threshold=MIN_THRESHOLD):
    """
    Automatically compute wet-day thresholds for observed and simulated data.
    
    Strategy:
    1. For observations: Use minimum non-zero value OR min_threshold, whichever is larger
       This respects instrument detection limits
    
    2. For simulations: Find the threshold that gives the same wet-day frequency
       as observations when applied to simulated data
    
    Parameters
    ----------
    obs_series : pd.Series
        Observed rainfall data
    sim_series : pd.Series  
        Simulated rainfall data
    min_threshold : float
        Minimum physical threshold (instrument detection limit)
    
    Returns
    -------
    tuple : (obs_threshold, sim_threshold, obs_wet_fraction, sim_wet_fraction_original)
    """
    print("\n" + "="*70)
    print("AUTOMATIC WET-DAY THRESHOLD COMPUTATION")
    print("="*70)
    
    # Observed data threshold
    obs_nonzero = obs_series[obs_series > 0].values
    if len(obs_nonzero) == 0:
        raise ValueError("No non-zero values in observed data!")
    
    obs_min_nonzero = obs_nonzero.min()
    obs_threshold = max(obs_min_nonzero, min_threshold)
    
    print(f"\nObserved Data:")
    print(f"  Total days: {len(obs_series)}")
    print(f"  Days with rain > 0: {len(obs_nonzero)} ({len(obs_nonzero)/len(obs_series)*100:.1f}%)")
    print(f"  Minimum non-zero: {obs_min_nonzero:.4f} mm")
    print(f"  Selected threshold: {obs_threshold:.4f} mm")
    
    # Calculate observed wet-day fraction
    obs_wet_count = np.sum(obs_series > obs_threshold)
    obs_wet_fraction = obs_wet_count / len(obs_series)
    
    print(f"  Wet days (> {obs_threshold:.4f} mm): {obs_wet_count} ({obs_wet_fraction*100:.1f}%)")
    
    # Simulated data - original statistics
    sim_nonzero = sim_series[sim_series > 0].values
    sim_wet_count_original = len(sim_nonzero)
    sim_wet_fraction_original = sim_wet_count_original / len(sim_series)
    
    print(f"\nERA5-Land Data (Original):")
    print(f"  Total days: {len(sim_series)}")
    print(f"  Days with rain > 0: {sim_wet_count_original} ({sim_wet_fraction_original*100:.1f}%)")
    
    # Find simulated threshold to match observed wet-day frequency
    # This is the key: we want sim_threshold such that:
    # sum(sim_series > sim_threshold) / len(sim_series) ≈ obs_wet_fraction
    
    if sim_wet_fraction_original < obs_wet_fraction:
        # Fewer wet days in sim than obs - unusual but possible
        print(f"\n  WARNING: Simulated has fewer wet days than observed!")
        print(f"  Using minimum threshold for simulated data.")
        sim_threshold = min_threshold
    else:
        # Normal case: sim has more wet days than obs (typical for GCM/reanalysis)
        # Find the value at (1 - obs_wet_fraction) percentile
        sim_threshold = np.percentile(sim_series, (1 - obs_wet_fraction) * 100)
        sim_threshold = max(sim_threshold, min_threshold)
    
    sim_wet_count_adjusted = np.sum(sim_series > sim_threshold)
    sim_wet_fraction_adjusted = sim_wet_count_adjusted / len(sim_series)
    
    print(f"\nERA5-Land Data (Adjusted):")
    print(f"  Selected threshold: {sim_threshold:.4f} mm")
    print(f"  Wet days (> {sim_threshold:.4f} mm): {sim_wet_count_adjusted} ({sim_wet_fraction_adjusted*100:.1f}%)")
    print(f"  Target wet-day fraction: {obs_wet_fraction*100:.1f}%")
    print(f"  Achieved wet-day fraction: {sim_wet_fraction_adjusted*100:.1f}%")
    print(f"  Difference: {abs(obs_wet_fraction - sim_wet_fraction_adjusted)*100:.2f}%")
    
    if abs(obs_wet_fraction - sim_wet_fraction_adjusted) > 0.02:
        print(f"\n  NOTE: Small discrepancy in wet-day frequency due to tied values at threshold")
    
    print("\n" + "="*70)
    
    return obs_threshold, sim_threshold, obs_wet_fraction, sim_wet_fraction_original


def fit_gamma_to_wet_days(series, threshold):
    """
    Fit a Gamma distribution to wet-day values only (val > threshold).
    Returns (shape, loc, scale, wet_day_fraction).
    loc is fixed at 0 — standard for rainfall.
    """
    wet = series[series > threshold].values
    if len(wet) < 10:
        raise ValueError(f"Too few wet days ({len(wet)}) to fit Gamma distribution.")
    
    # Fit gamma with loc=0 (standard for rainfall)
    shape, loc, scale = stats.gamma.fit(wet, floc=0)
    wet_frac = len(wet) / len(series)
    
    return shape, loc, scale, wet_frac


def gamma_quantile_mapping(obs, sim, obs_threshold, sim_threshold):
    """
    Quantile-to-Quantile bias correction using Gamma distribution mapping
    with automatic wet-day threshold determination.

    Algorithm:
      1. Fit Gamma to observed wet days (> obs_threshold)
      2. Identify simulated wet days (> sim_threshold) to keep
         Number kept = obs_wet_fraction × total_days
      3. Fit Gamma to kept simulated wet days
      4. Map each kept day via quantile transfer
      5. All other days remain zero

    Parameters
    ----------
    obs : pd.Series  
        Observed rainfall
    sim : pd.Series  
        Simulated rainfall to correct
    obs_threshold : float
        Wet-day threshold for observations
    sim_threshold : float
        Wet-day threshold for simulations (to match obs wet-day frequency)

    Returns
    -------
    np.ndarray — bias-corrected values (same length as sim)
    """
    print("\n" + "-"*70)
    print("GAMMA QUANTILE MAPPING")
    print("-"*70)
    
    # Fit gamma to observed wet days
    obs_shape, obs_loc, obs_scale, obs_wet_frac = fit_gamma_to_wet_days(obs, obs_threshold)
    
    print(f"\nObserved Gamma Distribution:")
    print(f"  Shape: {obs_shape:.4f}")
    print(f"  Scale: {obs_scale:.4f}")
    print(f"  Wet-day fraction: {obs_wet_frac:.4f}")
    
    corrected = np.zeros(len(sim))
    sim_vals  = sim.values
    all_idx   = np.arange(len(sim))
    
    # ── Step 1: Select wet days to keep ──────────────────────────────────────
    # Keep only values > sim_threshold (this matches obs wet-day frequency)
    keep_wet = all_idx[sim_vals > sim_threshold]
    n_kept = len(keep_wet)
    
    print(f"\nERA5-Land Days to Correct:")
    print(f"  Total days: {len(sim)}")
    print(f"  Days > {sim_threshold:.4f} mm: {n_kept} ({n_kept/len(sim)*100:.1f}%)")
    print(f"  Days to remain zero: {len(sim) - n_kept} ({(len(sim)-n_kept)/len(sim)*100:.1f}%)")
    
    if n_kept < 10:
        raise ValueError(f"Too few wet days to correct ({n_kept}). Check your thresholds.")
    
    # ── Step 2: Fit Gamma to kept simulated wet days ─────────────────────────
    subset_vals = sim_vals[keep_wet]
    
    # For fitting, we need values > 0
    # Since we already filtered by sim_threshold, use a minimal threshold
    sub_shape, sub_loc, sub_scale, _ = fit_gamma_to_wet_days(
        pd.Series(subset_vals), threshold=0.0  # all kept values are already wet
    )
    
    print(f"\nERA5-Land Gamma Distribution (kept subset):")
    print(f"  Shape: {sub_shape:.4f}")
    print(f"  Scale: {sub_scale:.4f}")
    print(f"  N samples: {len(subset_vals)}")
    
    # ── Step 3: Map each kept day via gamma quantile transfer ────────────────
    print(f"\nApplying quantile mapping to {n_kept} wet days...")
    
    for i, idx in enumerate(keep_wet):
        val = sim_vals[idx]
        
        # Quantile within the simulated distribution (kept subset)
        q = stats.gamma.cdf(val, sub_shape, sub_loc, sub_scale)
        
        # Clip to avoid numerical issues at extremes
        q = np.clip(q, 1e-6, 1 - 1e-6)
        
        # Map to observed distribution
        corrected[idx] = stats.gamma.ppf(q, obs_shape, obs_loc, obs_scale)
    
    # Check for any invalid values
    n_invalid = np.sum(np.isnan(corrected)) + np.sum(np.isinf(corrected))
    if n_invalid > 0:
        print(f"\n  WARNING: {n_invalid} invalid values generated, setting to 0")
        corrected[np.isnan(corrected)] = 0
        corrected[np.isinf(corrected)] = 0
    
    print(f"  Correction complete!")
    print("-"*70)
    
    return corrected


def print_statistics(label, series, threshold):
    """Print comprehensive distribution statistics for a rainfall series."""
    wet = series[series > threshold]
    
    print(f"\n{label}:")
    print(f"  Total days: {len(series)}")
    print(f"  Wet days (> {threshold:.4f} mm): {len(wet)} ({len(wet)/len(series)*100:.1f}%)")
    print(f"  Mean (all days): {series.mean():.4f} mm")
    print(f"  Std (all days): {series.std():.4f} mm")
    
    if len(wet) > 0:
        print(f"  Mean (wet days only): {wet.mean():.4f} mm")
        print(f"  Median (wet days): {wet.median():.4f} mm")
        print(f"  95th percentile (wet days): {wet.quantile(0.95):.4f} mm")
        print(f"  Max: {wet.max():.4f} mm")


def compute_r2_scores(obs, sim, corrected):
    """
    Compute R² for two comparison modes:
      - Point-to-point: direct daily value match (may be low for GCM data)
      - CDF-based:      quantile vs quantile (primary metric for QM)
    """
    min_len = min(len(obs), len(sim), len(corrected))
    o = obs.values[:min_len]
    s = sim[:min_len]
    c = corrected[:min_len]

    # Point-to-point R²
    r2_point_sim  = r2_score(o, s)
    r2_point_corr = r2_score(o, c)

    # CDF-based R²: compare sorted quantiles
    quantiles = np.linspace(0.01, 0.99, 200)
    obs_q  = np.quantile(o[o > 0], quantiles) if np.sum(o > 0) > 0 else np.zeros(200)
    sim_q  = np.quantile(s[s > 0], quantiles) if np.sum(s > 0) > 0 else np.zeros(200)
    corr_q = np.quantile(c[c > 0], quantiles) if np.sum(c > 0) > 0 else np.zeros(200)
    
    r2_cdf_sim  = r2_score(obs_q, sim_q) if np.sum(obs_q) > 0 else 0
    r2_cdf_corr = r2_score(obs_q, corr_q) if np.sum(obs_q) > 0 else 0

    return {
        "r2_point_sim":  r2_point_sim,
        "r2_point_corr": r2_point_corr,
        "r2_cdf_sim":    r2_cdf_sim,
        "r2_cdf_corr":   r2_cdf_corr,
    }


def plot_validation(df, obs, sim, corrected, r2, obs_threshold, sim_threshold, output_dir):
    """
        Four-panel validation figure:
      1. Q-Q plot (wet days only)
      2. PDF / histogram comparison
      3. CDF comparison
            4. Average monthly mean rainfall
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Gamma Quantile Mapping — Bias Correction Validation", 
                 fontsize=14, fontweight="bold")

    obs_arr  = obs.values
    sim_arr  = sim.values
    corr_arr = corrected

    # Wet-day subsets
    obs_wet  = np.sort(obs_arr[obs_arr   > obs_threshold])
    sim_wet  = np.sort(sim_arr[sim_arr   > sim_threshold])
    corr_wet = np.sort(corr_arr[corr_arr > obs_threshold])

    # ── Panel 1: Q-Q plot ────────────────────────────────────────────────────
    ax = axes[0, 0]
    qs = np.linspace(0.01, 0.99, 200)

    
    if len(obs_wet) > 0 and len(sim_wet) > 0:
        ax.plot(np.quantile(obs_wet, qs), np.quantile(sim_wet,  qs),
                "o", ms=4, alpha=0.5, label=f"Raw ERA5-Land (CDF R²={r2['r2_cdf_sim']:.3f})")
    if len(obs_wet) > 0 and len(corr_wet) > 0:
        ax.plot(np.quantile(obs_wet, qs), np.quantile(corr_wet, qs),
                "s", ms=4, alpha=0.5, label=f"Corrected (CDF R²={r2['r2_cdf_corr']:.3f})")
    
    lim = max( sim_wet.max() if len(sim_wet) > 0 else 1, 
              corr_wet.max() if len(corr_wet) > 0 else 1) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.5, label="1:1 line")
    ax.set_title("Q-Q Plot (wet days)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Observed quantiles (mm)", fontsize=10)
    ax.set_ylabel("Model quantiles (mm)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: PDF histogram ───────────────────────────────────────────────
    ax = axes[0, 1]
    all_wet = np.concatenate([obs_wet, sim_wet, corr_wet])
    if len(all_wet) > 0:
        upper = np.percentile(all_wet, 98)
        bins  = np.linspace(0, upper, 50)
        
        if len(obs_wet) > 0:
            ax.hist(obs_wet,  bins=bins, alpha=0.5, density=True, 
                   label=f"Observed (n={len(obs_wet)})", color='blue')
        if len(sim_wet) > 0:
            ax.hist(sim_wet,  bins=bins, alpha=0.5, density=True, 
                   label=f"ERA5-Land (n={len(sim_wet)})", color='orange')
        if len(corr_wet) > 0:
            ax.hist(corr_wet, bins=bins, alpha=0.5, density=True, 
                   label=f"Corrected (n={len(corr_wet)})", color='green')
    
    ax.set_title("PDF — Wet-Day Rainfall Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("Rainfall (mm)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: CDF comparison ──────────────────────────────────────────────
    ax = axes[1, 0]
    
    if len(obs_wet) > 0:
        obs_cdf = np.arange(1, len(obs_wet)+1) / len(obs_wet)
        ax.plot(obs_wet, obs_cdf, '-', linewidth=2, label='Observed', alpha=0.8)
    
    if len(sim_wet) > 0:
        sim_cdf = np.arange(1, len(sim_wet)+1) / len(sim_wet)
        ax.plot(sim_wet, sim_cdf, '--', linewidth=2, label='ERA5-Land', alpha=0.8)
    
    if len(corr_wet) > 0:
        corr_cdf = np.arange(1, len(corr_wet)+1) / len(corr_wet)
        ax.plot(corr_wet, corr_cdf, ':', linewidth=2, label='Corrected', alpha=0.8)
    
    ax.set_title("Cumulative Distribution Function (CDF)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Rainfall (mm)", fontsize=10)
    ax.set_ylabel("Cumulative Probability", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Average monthly mean rainfall ──────────────────────────────
    ax = axes[1, 1]
    date_col, dates = _find_date_column(df)

    if date_col is None:
        ax.text(0.5, 0.5, "Monthly mean plot skipped\n(no valid date column found)",
                ha="center", va="center", fontsize=10)
        ax.set_title("Average Monthly Mean Rainfall", fontsize=11, fontweight="bold")
        ax.axis("off")
    else:
        monthly_df = df.copy()
        monthly_df["_date"] = dates
        monthly_df = monthly_df[monthly_df["_date"].notna()].copy()
        monthly_df["_month"] = monthly_df["_date"].dt.month

        monthly_mean = monthly_df.groupby("_month")[[OBS_COL, SIM_COL, OUT_COL]].mean()
        monthly_mean = monthly_mean.reindex(range(1, 13))

        ax.plot(monthly_mean.index, monthly_mean[OBS_COL], marker="o", linewidth=2,
                color="tab:blue", label="Observed")
        ax.plot(monthly_mean.index, monthly_mean[SIM_COL], marker="s", linewidth=2,
                color="tab:orange", label="ERA5-Land (tp_rainfall)")
        ax.plot(monthly_mean.index, monthly_mean[OUT_COL], marker="^", linewidth=2,
                color="tab:green", label="Corrected")

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.set_title("Average Monthly Mean Rainfall", fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=10)
        ax.set_ylabel("Rainfall (mm/day)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "bias_correction_validation.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nValidation plot saved → {out_path}")
    plt.show()


def _find_date_column(df):
    """Find a usable date column for monthly aggregation."""
    date_candidates = ["date", "Date", "DATE", "datetime", "Datetime", "time", "Time"]

    for col in date_candidates:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0:
                return col, parsed

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col, pd.to_datetime(df[col], errors="coerce")

    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().sum() > 0.8 * len(df):
            return col, parsed

    return None, None


def plot_monthly_mean_rainfall(df):
    """Plot average monthly mean rainfall for observed, ERA5-Land, and corrected data."""
    date_col, dates = _find_date_column(df)
    if date_col is None:
        print("\nMonthly mean plot skipped: no valid date column found.")
        return

    monthly_df = df.copy()
    monthly_df["_date"] = dates
    monthly_df = monthly_df[monthly_df["_date"].notna()].copy()

    monthly_df["_month"] = monthly_df["_date"].dt.month

    monthly_mean = monthly_df.groupby("_month")[[OBS_COL, SIM_COL, OUT_COL]].mean()
    monthly_mean = monthly_mean.reindex(range(1, 13))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly_mean.index, monthly_mean[OBS_COL], marker="o", linewidth=2,
            color="tab:blue", label="Observed")
    ax.plot(monthly_mean.index, monthly_mean[SIM_COL], marker="s", linewidth=2,
            color="tab:orange", label="ERA5-Land (tp_rainfall)")
    ax.plot(monthly_mean.index, monthly_mean[OUT_COL], marker="^", linewidth=2,
            color="tab:green", label="Corrected")

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title("Average Monthly Mean Rainfall", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rainfall (mm/day)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_path = "monthly_mean_rainfall_comparison.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Monthly mean rainfall plot saved → {out_path} (date column: {date_col})")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("GAMMA QUANTILE MAPPING WITH AUTOMATIC WET-DAY THRESHOLD")
    print("="*70)
    
    print("\nLoading data...")
    df = load_data(DATA_FILE)
    print(f"  {len(df)} rows loaded")
    print(f"  Columns: {list(df.columns)}")

    # Compute wet-day thresholds automatically
    obs_threshold, sim_threshold, obs_wet_frac, sim_wet_frac_orig = \
        compute_wet_day_threshold(df[OBS_COL], df[SIM_COL], min_threshold=MIN_THRESHOLD)

    print("\n" + "="*70)
    print("DISTRIBUTION STATISTICS (BEFORE CORRECTION)")
    print("="*70)
    print_statistics("Observed", df[OBS_COL], obs_threshold)
    print_statistics("ERA5-Land", df[SIM_COL], sim_threshold)

    print("\n" + "="*70)
    print("APPLYING BIAS CORRECTION")
    print("="*70)
    
    corrected = gamma_quantile_mapping(
        df[OBS_COL], 
        df[SIM_COL], 
        obs_threshold, 
        sim_threshold
    )

    df[OUT_COL] = corrected

    print("\n" + "="*70)
    print("DISTRIBUTION STATISTICS (AFTER CORRECTION)")
    print("="*70)
    print_statistics("Corrected", df[OUT_COL], obs_threshold)

    print("\n" + "="*70)
    print("VALIDATION METRICS")
    print("="*70)
    
    r2 = compute_r2_scores(df[OBS_COL], df[SIM_COL], corrected)
    
    print(f"\nR² Scores:")
    print(f"  Point-to-point (daily values):")
    print(f"    Raw ERA5-Land:  {r2['r2_point_sim']:.4f}")
    print(f"    Bias-corrected: {r2['r2_point_corr']:.4f}")
    print(f"    Improvement:    {r2['r2_point_corr'] - r2['r2_point_sim']:+.4f}")
    print(f"\n  CDF-based (quantiles) - PRIMARY METRIC FOR QM:")
    print(f"    Raw ERA5-Land:  {r2['r2_cdf_sim']:.4f}")
    print(f"    Bias-corrected: {r2['r2_cdf_corr']:.4f}")
    print(f"    Improvement:    {r2['r2_cdf_corr'] - r2['r2_cdf_sim']:+.4f}")
    
    # Wet-day frequency check
    corr_wet_frac = np.sum(df[OUT_COL] > obs_threshold) / len(df[OUT_COL])
    print(f"\nWet-Day Frequency Check:")
    print(f"  Observed:       {obs_wet_frac*100:.2f}%")
    print(f"  ERA5-Land (raw): {sim_wet_frac_orig*100:.2f}%")
    print(f"  Corrected:      {corr_wet_frac*100:.2f}%")
    print(f"  Match quality:  {abs(obs_wet_frac - corr_wet_frac)*100:.2f}% difference")

    print("\n" + "="*70)
    print("GENERATING VALIDATION PLOTS")
    print("="*70)

    out_file = r"D:\Masters\data\predicting_flashflood_events_codes\era5_corrected_output.csv"
    output_dir = os.path.dirname(out_file)
    plot_validation(df, df[OBS_COL], df[SIM_COL], corrected, r2, obs_threshold, sim_threshold, output_dir)

    df.to_csv(out_file, index=False)
    print(f"\n" + "="*70)
    print(f"Output saved → {out_file}")
    print("="*70)

    # ── Save fitted parameters for use by apply_correction.py ────────────────
    obs_shape, obs_loc, obs_scale, obs_wet_frac_fit = fit_gamma_to_wet_days(
        df[OBS_COL], obs_threshold
    )
    params = {
        "obs_threshold":    float(obs_threshold),
        "sim_threshold":    float(sim_threshold),
        "obs_gamma_shape":  float(obs_shape),
        "obs_gamma_loc":    float(obs_loc),
        "obs_gamma_scale":  float(obs_scale),
        "obs_wet_fraction": float(obs_wet_frac),
        "era5_col":         SIM_COL,
        "out_col":          OUT_COL,
    }
    params_file = os.path.join(output_dir, "gamma_qm_params.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Fitted parameters saved → {params_file}")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  Observed wet-day threshold: {obs_threshold:.4f} mm")
    print(f"  ERA5-Land wet-day threshold: {sim_threshold:.4f} mm")
    print(f"  Wet-day frequency matched: {obs_wet_frac*100:.1f}%")
    print(f"  CDF R² improvement: {r2['r2_cdf_sim']:.3f} → {r2['r2_cdf_corr']:.3f}")
    print(f"  Output column: '{OUT_COL}'")


if __name__ == "__main__":
    main()
