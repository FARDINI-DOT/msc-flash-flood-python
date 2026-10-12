"""
Run Storm Event Analysis
========================
This script loads your Excel data and identifies storm events.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from storm_event_analyzer_mergedv3 import StormEventAnalyzer


def compute_rejection_breakdown(analyzer):
    """Compute where candidate events are filtered out by current criteria."""
    df = analyzer.df
    outer_counts = Counter()
    inner_counts = Counter()

    def evaluate_candidate(rain_start_idx):
        precip_end_idx = rain_start_idx
        consecutive_dry_count = 0
        max_dry_gap_encountered = 0

        for j in range(rain_start_idx, min(rain_start_idx + analyzer.max_precip_spell_duration_hours, len(df))):
            current_precip = df.iloc[j][analyzer.precip_col]

            if current_precip >= analyzer.precip_spell_end_threshold:
                precip_end_idx = j
                consecutive_dry_count = 0
            else:
                consecutive_dry_count += 1
                max_dry_gap_encountered = max(max_dry_gap_encountered, consecutive_dry_count)

            if j >= rain_start_idx + analyzer.spell_dry_window_hours - 1:
                rolling_total = df.iloc[j-analyzer.spell_dry_window_hours+1:j+1][analyzer.precip_col].sum()
                if rolling_total <= analyzer.spell_dry_total_threshold:
                    break

        if max_dry_gap_encountered >= analyzer.max_dry_gap_within_spell:
            return None, "multiple_rainfall_blocks"

        total_precip = df.iloc[rain_start_idx:precip_end_idx+1][analyzer.precip_col].sum()
        if total_precip < analyzer.min_total_precip:
            return None, "total_precip_below_min"

        start_window = df.iloc[rain_start_idx:precip_end_idx+1]
        start_idx = start_window[analyzer.wl_col].idxmin()
        initial_wl = df.iloc[start_idx][analyzer.wl_col]

        wl_has_risen = False
        return_idx = None
        peak_wl = initial_wl

        for j in range(start_idx, min(start_idx + 240, len(df))):
            current_wl = df.iloc[j][analyzer.wl_col]

            if current_wl > peak_wl:
                peak_wl = current_wl

            if current_wl > initial_wl + analyzer.return_tolerance:
                wl_has_risen = True

            if wl_has_risen and current_wl <= initial_wl + analyzer.event_end_threshold:
                return_idx = j
                break

        if not wl_has_risen:
            return None, "wl_rise_below_min"

        if return_idx is None:
            return None, "wl_not_returned_to_baseline"

        # Match analyzer rule: reject rise-fall-rise hydrographs.
        event_wl = df.iloc[start_idx:return_idx+1][analyzer.wl_col].to_numpy()
        wl_diff = np.diff(event_wl)
        wl_states = []
        for d in wl_diff:
            if d > analyzer.wl_reversal_tolerance:
                wl_states.append(1)
            elif d < -analyzer.wl_reversal_tolerance:
                wl_states.append(-1)

        compressed_states = []
        for s in wl_states:
            if not compressed_states or compressed_states[-1] != s:
                compressed_states.append(s)

        if -1 in compressed_states:
            first_fall_idx = compressed_states.index(-1)
            if 1 in compressed_states[first_fall_idx + 1:]:
                return None, "multiple_rise_fall_cycles"

        return {"start_idx": start_idx, "end_idx": return_idx}, "passed"

    i = analyzer.baseflow_stability_hours
    while i < len(df) - 24:
        antecedent_window = df.iloc[i-analyzer.dry_period_hours:i]
        if antecedent_window[analyzer.precip_col].max() > analyzer.antecedent_dry_threshold:
            outer_counts["antecedent_not_dry"] += 1
            i += 1
            continue

        baseflow_window = df.iloc[i-analyzer.baseflow_stability_hours:i]
        wl_variation = baseflow_window[analyzer.wl_col].max() - baseflow_window[analyzer.wl_col].min()
        if wl_variation > analyzer.baseflow_variation_threshold:
            outer_counts["baseflow_unstable"] += 1
            i += 1
            continue

        if df.iloc[i][analyzer.precip_col] <= analyzer.precip_event_start_threshold:
            outer_counts["rain_trigger_not_met"] += 1
            i += 1
            continue

        event, reason = evaluate_candidate(i)
        inner_counts[reason] += 1
        if event is not None:
            i = event["end_idx"] + analyzer.dry_period_hours
        else:
            i += 1

    return outer_counts, inner_counts

# Load your data
print("Loading data from Excel...")
# TODO: Replace with your actual file path
file_path = r"D:\Masters\data\predicting_flashflood_events_codes\PREMONSOON-ALL-FILLEDv5_corrected.xlsx"  # UPDATE THIS
df = pd.read_excel(file_path)

print(f"Data loaded: {len(df)} records")
print(f"Date range: {df['Date Time'].min()} to {df['Date Time'].max()}")
print("\nFirst few rows:")
print(df.head())

# Initialize the analyzer
analyzer = StormEventAnalyzer(
    df, 
    wl_col='WL (mMSL)', 
    precip_col='corr_precip', 
    datetime_col='Date Time',
    q_col='Q'
)

# Diagnostic: where events are being filtered out the most.
outer_counts, inner_counts = compute_rejection_breakdown(analyzer)
print("\n" + "="*60)
print("EVENT REJECTION BREAKDOWN")
print("="*60)
print("Outer filters (scan-level):")
for k, v in outer_counts.most_common():
    print(f"- {k}: {v}")
print("Inner filters (candidate-event level):")
for k, v in inner_counts.most_common():
    print(f"- {k}: {v}")

# Identify storm events
events = analyzer.identify_storm_events()

if len(events) > 0:
    # Create summary table
    print("\n" + "="*60)
    print("STORM EVENT SUMMARY")
    print("="*60)
    summary = analyzer.create_summary_table()
    if summary is None:
        raise RuntimeError("Summary creation failed after events were identified.")
    print(summary.to_string(index=False))
    
    # Save summary to Excel
    summary.to_excel("D:\\Masters\\data\\predicting_flashflood_events_codes\\storm_events_summary.xlsx", index=False)
    print("\nSummary saved to: D:\\Masters\\data\\predicting_flashflood_events_codes\\storm_events_summary.xlsx")
    
    # Calculate statistics
    print("\n" + "="*60)
    print("TIME TO PEAK STATISTICS")
    print("="*60)
    stats = analyzer.get_time_to_peak_statistics()
    if stats is None:
        raise RuntimeError("Statistics calculation failed after events were identified.")
    print(stats)
    
    # Plot first 3 events as examples
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    # Plot first few events individually
    for i in range(min(3, len(events))):
        print(f"\nPlotting Event {i+1}...")
        fig = analyzer.plot_event(i+1)
        plt.savefig(rf"D:\Masters\data\predicting_flashflood_events_codes\event_{i+1}_example.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate all event plots
    print("\nGenerating all event plots...")
    analyzer.plot_all_events(output_dir=r"D:\Masters\data\predicting_flashflood_events_codes\all_storm_events")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Total events identified: {len(events)}")
    print(f"Average time to peak: {stats.loc['mean', 'Time_to_Peak']:.2f} hours")
    print(f"Average lag time: {stats.loc['mean', 'Lag_Time']:.2f} hours")
    print("\nFiles created:")
    print("- storm_events_summary.xlsx (event details)")
    print("- all_storm_events/ (individual event plots)")
else:
    print("\nNo storm events found with the current criteria.")
    print("You may need to adjust the parameters.")
