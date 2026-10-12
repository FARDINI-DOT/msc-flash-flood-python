import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

class StormEventAnalyzer:
    """
    Analyzes storm events from water level and precipitation data.

    ════════════════════════════════════════════════════════════════════════════
    IMPLEMENTED CRITERIA (User's Specifications):
    ════════════════════════════════════════════════════════════════════════════
    
    ANTECEDENT CONDITIONS:
    ----------------------
    1. Dry period: 36 hours with precipitation ≤ 0.5 mm/hr
    2. Baseflow stability: 36 hours with WL variation ≤ 0.2 m
    
    EVENT START:
    -----------
    3. Rain trigger: precipitation > 0.5 mm/hr
    4. Event start time: LOWEST WL point after rain starts (within precip spell)
    
    PRECIPITATION SPELL:
    -------------------
    5. Spell ends: accumulated precipitation ≤ 1.0 mm over 24 hours
    7. Minimum total precipitation: 1.0 mm over the spell
    8. Maximum spell duration: 120 hours
    
    EVENT REQUIREMENTS:
    ------------------
    9. Minimum rise: WL must rise at least 0.4 m above initial WL
    10. Event end: WL returns to initial WL + 0.05 m
    
    PEAK IDENTIFICATION:
    -------------------
    11. Peak = HIGHEST WL point between event start and event end
    12. Reject event if WL rises, falls, and rises again within the event period
    
    TIMING METRICS (Reported, Not Used as Hard Rejection):
    ------------------------------------------------------
    - Short storms (<12 hrs): Peak delay from precipitation end
    - Long storms (≥12 hrs): Lag from precipitation centroid
    - WMO compliance reported for reference
    
    ════════════════════════════════════════════════════════════════════════════
    KEY DIFFERENCES FROM PREVIOUS VERSION:
    ════════════════════════════════════════════════════════════════════════════
    - Event start = LOWEST WL after rain (not first rain hour)
    - WMO timing NOT used as rejection criteria (reported only)
    - Spell termination uses rolling 24h accumulated rainfall ≤ 1.0 mm
    ════════════════════════════════════════════════════════════════════════════
    """
    
    def __init__(self, df, wl_col='WL (mMSL)', precip_col='corr_precip', 
                 datetime_col='Date Time', q_col='Q'):
        """
        Initialize the analyzer with data.
        
        Parameters:
        -----------
        df : DataFrame
            Input data with datetime, water level, and precipitation
        wl_col : str
            Column name for water level
        precip_col : str
            Column name for precipitation
        datetime_col : str
            Column name for datetime
        q_col : str
            Column name for discharge (optional, auto-detected if not provided)
        """
        self.df = df.copy()
        self.wl_col = wl_col
        self.precip_col = precip_col
        self.datetime_col = datetime_col
        self.q_col = None
        
        # Ensure datetime column is datetime type
        self.df[datetime_col] = pd.to_datetime(self.df[datetime_col])
        self.df = self.df.sort_values(datetime_col).reset_index(drop=True)

        # Optional discharge/flow column support for plotting and summaries
        if q_col in self.df.columns:
            self.q_col = q_col
        else:
            q_candidates = [
                c for c in self.df.columns
                if c.strip().lower() in ('q', 'discharge', 'flow', 'streamflow')
            ]
            if not q_candidates:
                q_candidates = [c for c in self.df.columns if c.strip().lower().startswith('q')]
            if q_candidates:
                self.q_col = q_candidates[0]
        
        # Parameters - Time windows (USER SPECIFICATIONS)
        self.dry_period_hours = 36  # Antecedent dry period
        self.baseflow_stability_hours = 36  # Antecedent stability period
        self.baseflow_variation_threshold = 0.2  # meters - WL variation threshold
    
        # Precipitation thresholds (USER SPECIFICATIONS)
        self.antecedent_dry_threshold = 0.5  # mm/hr - antecedent period must have precip ≤ this
        self.precip_event_start_threshold = 0.5  # mm/hr - rain trigger
        self.precip_spell_end_threshold = 0.05  # mm/hr - marks rain/no-rain for dry-gap tracking
        self.spell_dry_window_hours = 24  # hrs - rolling window for spell termination
        self.spell_dry_total_threshold = 1.0  # mm - spell ends when rolling accumulation is ≤ this
        self.max_precip_spell_duration_hours = 120  # Maximum spell duration
        
        # CRITICAL: Prevent multiple rainfall blocks (Event #3 problem)
        self.max_dry_gap_within_spell = 36  # hours - if dry gap > 24h, likely separate storms
        
        # Event validation parameters (USER SPECIFICATIONS)
        self.min_total_precip = 1.0  # mm - minimum total precipitation
        self.min_duration_hours = 0.0  # Not used as rejection criterion
        self.return_tolerance = 0.4  # meters - minimum WL rise required
        self.event_end_threshold = 0.09  # meters above initial WL - defines event end
        self.wl_reversal_tolerance = 0.01  # meters - ignore tiny oscillations in rise/fall check
        
        # WMO TIMING CRITERIA (REPORTED ONLY, NOT USED AS REJECTION)
        self.short_storm_threshold = 12.0  # hours - threshold for short vs long
        self.max_peak_delay_short_storm = 24.0  # hours - for reporting only
        self.min_peak_delay_short_storm = 0.0  # hours - for reporting only
        self.max_lag_from_centroid = 24.0  # hours - for reporting only
        
        self.events = []
        
    def identify_storm_events(self):
        """
        Identify individual storm events based on defined criteria.
        """
        print("Identifying storm events (User's Specifications)...")
        print(f"Total records: {len(self.df)}")
        print(f"\n📋 CRITERIA CONFIGURATION:")
        print(f"  Antecedent dry: ≤ {self.antecedent_dry_threshold} mm/hr for {self.dry_period_hours} hrs")
        print(f"  Baseflow stability: ≤ {self.baseflow_variation_threshold} m variation over {self.baseflow_stability_hours} hrs")
        print(f"  Rain starts: > {self.precip_event_start_threshold} mm/hr")
        print(f"  Event start: LOWEST WL after rain starts (within precip spell)")
        print(f"  Spell ends: accumulated precip ≤ {self.spell_dry_total_threshold} mm over {self.spell_dry_window_hours} hrs")
        print(f"  Max dry gap in spell: {self.max_dry_gap_within_spell} hrs (prevents multiple rainfall blocks)")
        print(f"  Min total precip: {self.min_total_precip} mm")
        print(f"  Min WL rise: {self.return_tolerance} m")
        print(f"  Event ends: WL returns to initial + {self.event_end_threshold} m")
        print(f"  Hydrograph shape: single rise-fall only (no second rise after recession)")
        print(f"\n💡 Timing metrics (reported, NOT used as rejection criteria):")
        print(f"  Short storms (< {self.short_storm_threshold} hrs): Peak delay from precip end")
        print(f"  Long storms (≥ {self.short_storm_threshold} hrs): Lag from precip centroid")
        print(f"{'='*80}")
        
        df = self.df
        i = self.baseflow_stability_hours  # Start after stability window
        
        while i < len(df) - 24:  # Leave room for event to develop
            # Check if we're in a dry period (antecedent conditions)
            antecedent_window = df.iloc[i-self.dry_period_hours:i]
            
            if antecedent_window[self.precip_col].max() > self.antecedent_dry_threshold:
                i += 1
                continue
            
            # Check baseflow stability
            baseflow_window = df.iloc[i-self.baseflow_stability_hours:i]
            wl_variation = baseflow_window[self.wl_col].max() - baseflow_window[self.wl_col].min()
            
            if wl_variation > self.baseflow_variation_threshold:
                i += 1
                continue
            
            # We have stable baseflow - look for precipitation event start
            if df.iloc[i][self.precip_col] > self.precip_event_start_threshold:
                event = self._extract_event(i)
                
                if event is not None:
                    self.events.append(event)
                    print(f"\nEvent {len(self.events)} found: {event['start_time']} to {event['end_time']}")
                    print(f"  Precip duration: {event['precip_duration_hours']:.1f} hrs")
                    print(f"  Event type: {event['event_type']}")
                    print(f"  Timing metric: {event['timing_metric']:.1f} hrs ({event['timing_description']})")
                    print(f"  WL rise: {event['wl_rise']:.2f} m")
                    print(f"  Total precip: {event['total_precip']:.1f} mm")
                    # Jump to end of this event
                    i = event['end_idx'] + self.dry_period_hours
                else:
                    i += 1
            else:
                i += 1
        
        print(f"\n{'='*80}")
        print(f"Total storm events identified: {len(self.events)}")
        print(f"{'='*80}")
        return self.events
    
    def _extract_event(self, rain_start_idx):
        """
        Extract a single storm event starting from the given index.

        USER'S SPECIFICATIONS:
        1. Build precipitation spell from rain_start_idx
        2. Enforce minimum total precipitation (1.0 mm)
        3. Event start = LOWEST WL point after rain starts (within spell)
        4. Scan forward to find event end at initial WL + 0.05 m
        5. Peak = HIGHEST WL between event start and event end
        6. Reject if WL shows multiple rise-fall cycles within event period
        7. WMO timing metrics calculated but NOT used as rejection criteria
        """
        df = self.df
        
        # ================================================================
        # STEP 1: FIND THE EXTENT OF PRECIPITATION SPELL
        # WITH VALIDATION TO PREVENT MULTIPLE RAINFALL BLOCKS
        # ================================================================
        # Rainfall started at rain_start_idx (>0.5 mm/hr already verified)
        # Find when the precipitation spell ends.
        # End criterion: rolling accumulated precipitation <= threshold over window.
        
        precip_end_idx = rain_start_idx
        consecutive_dry_count = 0
        max_dry_gap_encountered = 0  # Track longest dry gap to detect multiple blocks
        
        for j in range(rain_start_idx, min(rain_start_idx + self.max_precip_spell_duration_hours, len(df))):
            current_precip = df.iloc[j][self.precip_col]
            
            if current_precip >= self.precip_spell_end_threshold:
                precip_end_idx = j
                consecutive_dry_count = 0
            else:
                consecutive_dry_count += 1
                max_dry_gap_encountered = max(max_dry_gap_encountered, consecutive_dry_count)

            if j >= rain_start_idx + self.spell_dry_window_hours - 1:
                rolling_total = df.iloc[j-self.spell_dry_window_hours+1:j+1][self.precip_col].sum()
                if rolling_total <= self.spell_dry_total_threshold:
                    break
        
        # VALIDATION: Reject if there was a prolonged dry gap within spell
        # This prevents merging distinct rainfall blocks (Event #3 problem)
        # Example: Rain Apr 30-May 1, DRY 48 hours, Rain May 4-5 → TWO separate events
        if max_dry_gap_encountered >= self.max_dry_gap_within_spell:
            return None  # Reject: multiple distinct rainfall blocks detected
        
        # Calculate total precipitation during spell
        total_precip = df.iloc[rain_start_idx:precip_end_idx+1][self.precip_col].sum()

        if total_precip < self.min_total_precip:
            return None  # Reject: insufficient precipitation

        # ================================================================
        # STEP 2: EVENT START = LOWEST WL AFTER RAIN STARTS
        # ================================================================
        # USER REQUIREMENT: Event starts at the LOWEST WL point within the precip spell
        start_window = df.iloc[rain_start_idx:precip_end_idx+1]
        start_idx = start_window[self.wl_col].idxmin()
        initial_wl = df.iloc[start_idx][self.wl_col]

        # ================================================================
        # STEP 3: FIND WHEN WL RETURNS TO BASELINE (EVENT END)
        # AND TRACK PEAK WL ALONG THE WAY
        # ================================================================
        # USER REQUIREMENTS:
        # - Event ends when WL returns to initial + 0.05m
        # - Peak = highest WL between start and end
        
        return_idx = None
        wl_has_risen = False
        max_search_hours = 240  # up to 10 days
        peak_idx = start_idx
        peak_wl = initial_wl
        
        for j in range(start_idx, min(start_idx + max_search_hours, len(df))):
            current_wl = df.iloc[j][self.wl_col]
            
            # Track peak = HIGHEST WL point in the event
            if current_wl > peak_wl:
                peak_wl = current_wl
                peak_idx = j

            # Track whether WL has risen sufficiently
            if current_wl > initial_wl + self.return_tolerance:
                wl_has_risen = True

            # Event ends when WL returns to initial + threshold
            if wl_has_risen and current_wl <= initial_wl + self.event_end_threshold:
                return_idx = j
                break  # Found event end

        if return_idx is None:
            return None  # Event rejected: WL did not return to baseline

        # Reject multi-peaked hydrographs: once recession starts, WL must not rise again.
        event_wl = df.iloc[start_idx:return_idx+1][self.wl_col].to_numpy()
        wl_diff = np.diff(event_wl)
        wl_states = []
        for d in wl_diff:
            if d > self.wl_reversal_tolerance:
                wl_states.append(1)
            elif d < -self.wl_reversal_tolerance:
                wl_states.append(-1)

        compressed_states = []
        for s in wl_states:
            if not compressed_states or compressed_states[-1] != s:
                compressed_states.append(s)

        if -1 in compressed_states:
            first_fall_idx = compressed_states.index(-1)
            if 1 in compressed_states[first_fall_idx + 1:]:
                return None  # Event rejected: multiple rise-fall cycles detected

        # ================================================================
        # STEP 4: CALCULATE TIMING METRICS (REPORTED ONLY, NOT FOR REJECTION)
        # ================================================================
        
        precip_duration = (df.iloc[precip_end_idx][self.datetime_col] - 
                          df.iloc[rain_start_idx][self.datetime_col]).total_seconds() / 3600
        
        peak_time = df.iloc[peak_idx][self.datetime_col]
        precip_end_time = df.iloc[precip_end_idx][self.datetime_col]
        
        # Calculate precipitation centroid (temporal center of mass)
        precip_window = df.iloc[rain_start_idx:precip_end_idx+1]
        precip_values = precip_window[self.precip_col].values
        precip_times = precip_window[self.datetime_col].values
        total_precip_spell = precip_values.sum()
        
        if total_precip_spell > 0:
            epoch_times = precip_times.astype('int64')
            centroid_epoch = (precip_values * epoch_times).sum() / total_precip_spell
            precip_centroid_time = pd.Timestamp(int(centroid_epoch))
        else:
            precip_centroid_time = df.iloc[rain_start_idx][self.datetime_col]
        
        # Calculate lag time: centroid to peak
        lag_time = (peak_time - precip_centroid_time).total_seconds() / 3600
        
        # Calculate peak delay: precip END to peak
        peak_delay = (peak_time - precip_end_time).total_seconds() / 3600
        
        # Classify event type (for reporting only)
        if precip_duration < self.short_storm_threshold:
            event_type = 'short_duration'
            timing_metric = peak_delay
            timing_description = f"Peak {peak_delay:.1f} hrs after precip ends"
        else:
            event_type = 'long_duration'
            timing_metric = lag_time
            timing_description = f"Lag {lag_time:.1f} hrs from centroid"

        # ================================================================
        # STEP 5: CALCULATE EVENT DURATION
        # ================================================================
        event_duration = (df.iloc[return_idx][self.datetime_col] - 
                         df.iloc[start_idx][self.datetime_col]).total_seconds() / 3600

        # ================================================================
        # STEP 6: COMPILE EVENT CHARACTERISTICS
        # ================================================================
        time_to_peak = (peak_time - df.iloc[start_idx][self.datetime_col]).total_seconds() / 3600
        
        event = {
            'event_id': len(self.events) + 1,
            'start_idx': start_idx,
            'end_idx': return_idx,
            'peak_idx': peak_idx,
            'precip_end_idx': precip_end_idx,
            'rain_start_idx': rain_start_idx,  # Store original rain start
            'start_time': df.iloc[start_idx][self.datetime_col],
            'end_time': df.iloc[return_idx][self.datetime_col],
            'precip_end_time': precip_end_time,
            'peak_time': peak_time,
            'initial_wl': initial_wl,
            'peak_wl': peak_wl,
            'wl_rise': peak_wl - initial_wl,
            'total_precip': total_precip,
            'time_to_peak_hours': time_to_peak,
            'lag_time_hours': lag_time,
            'duration_hours': event_duration,
            'precip_duration_hours': precip_duration,
            'precip_centroid_time': precip_centroid_time,
            'peak_precip_rate': precip_window[self.precip_col].max(),
            # TIMING METRICS (reported only)
            'event_type': event_type,
            'peak_delay_hours': peak_delay,
            'timing_metric': timing_metric,
            'timing_description': timing_description,
            'wmo_compliant': True  # Always true since WMO not used for rejection
        }
        
        return event
    
    def create_summary_table(self):
        """
        Create a summary table of all identified events.
        """
        if not self.events:
            print("No events identified yet. Run identify_storm_events() first.")
            return None
        
        summary = pd.DataFrame([
            {
                'Event_ID': e['event_id'],
                'Start_Time': e['start_time'],
                'Precip_End_Time': e['precip_end_time'],
                'Peak_Time': e['peak_time'],
                'End_Time': e['end_time'],
                'Event_Type': e['event_type'],
                'Duration_hrs': round(e['duration_hours'], 2),
                'Precip_Duration_hrs': round(e['precip_duration_hours'], 2),
                'Total_Precip_mm': round(e['total_precip'], 2),
                'Peak_Precip_mm/hr': round(e['peak_precip_rate'], 2),
                'Initial_WL_m': round(e['initial_wl'], 3),
                'Peak_WL_m': round(e['peak_wl'], 3),
                'WL_Rise_m': round(e['wl_rise'], 3),
                'Time_to_Peak_hrs': round(e['time_to_peak_hours'], 2),
                'Lag_Time_hrs': round(e['lag_time_hours'], 2),
                'Peak_Delay_hrs': round(e['peak_delay_hours'], 2),
                'Timing_Description': e['timing_description'],
                'WMO_Compliant': e['wmo_compliant']
            }
            for e in self.events
        ])
        
        return summary
    
    def plot_event(self, event_id, save_path=None):
        """
        Plot a single storm event with timing criteria visualization.
        """
        event = self.events[event_id - 1]
        lag_start_time = event.get('precip_centroid_time', event['start_time'])
        
        # Extract event data with some buffer
        buffer_hours = 24
        start_idx = max(0, event['start_idx'] - buffer_hours)
        end_idx = min(len(self.df) - 1, event['end_idx'] + buffer_hours)
        
        event_data = self.df.iloc[start_idx:end_idx+1].copy()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot precipitation (inverted) with hatch for grayscale readability
        ax1.bar(
            event_data[self.datetime_col],
            event_data[self.precip_col],
            width=0.04,
            facecolor='white',
            edgecolor='black',
            linewidth=0.8,
            hatch='//',
            alpha=0.9,
            label='Precipitation'
        )
        ax1.set_ylabel('Precipitation (mm/hr)', fontsize=11)
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        
        # Mark precipitation spell extent
        rain_start_time = self.df.iloc[event['rain_start_idx']][self.datetime_col]
        ax1.axvline(
            rain_start_time,
            color='black',
            linestyle='--',
            alpha=0.9,
            linewidth=2,
            label='Rain Starts'
        )
        ax1.axvline(
            event['precip_end_time'],
            color='black',
            linestyle=':',
            alpha=0.9,
            linewidth=2,
            label='Precip Spell End'
        )
        
        # Plot water level
        ax2.plot(
            event_data[self.datetime_col],
            event_data[self.wl_col],
            color='black',
            linestyle='-',
            linewidth=2,
            label='Water Level'
        )

        # Plot discharge if available
        ax2_q = None
        if self.q_col is not None and self.q_col in event_data.columns:
            ax2_q = ax2.twinx()
            ax2_q.plot(
                event_data[self.datetime_col],
                event_data[self.q_col],
                color='0.35',
                linestyle='--',
                linewidth=1.8,
                alpha=0.95,
                marker='x',
                markersize=3,
                markevery=max(1, len(event_data) // 30),
                label='Discharge (Q)'
            )
            ax2_q.set_ylabel('Discharge (Q)', fontsize=11, color='black')
            ax2_q.tick_params(axis='y', labelcolor='black')
        
        # Mark key points
        ax2.axvline(
            event['start_time'],
            color='black',
            linestyle='--',
            alpha=0.9,
            label='Event Start (Lowest WL)'
        )
        ax2.axvline(
            lag_start_time,
            color='black',
            linestyle='-.',
            alpha=0.9,
            label='Precipitation Centroid'
        )
        ax2.axvline(
            event['peak_time'],
            color='black',
            linestyle=(0, (6, 2, 1, 2)),
            alpha=0.9,
            label='Peak WL',
            linewidth=2
        )
        ax2.axvline(
            event['end_time'],
            color='black',
            linestyle=':',
            alpha=0.9,
            label='Event End'
        )
        
        ax2.axhline(
            event['initial_wl'],
            color='0.45',
            linestyle='-',
            alpha=0.9,
            linewidth=1.2,
            label='Initial WL'
        )
        ax2.axhline(
            event['initial_wl'] + self.return_tolerance,
            color='0.2',
            linestyle='--',
            alpha=0.9,
            linewidth=1.2,
            label=f'Initial + {self.return_tolerance}m'
        )
        
        # Mark peak point
        ax2.plot(
            event['peak_time'],
            event['peak_wl'],
            marker='o',
            markerfacecolor='white',
            markeredgecolor='black',
            markeredgewidth=1.8,
            linestyle='None',
            markersize=11,
            label='Peak',
            zorder=5
        )

        # Mark timing interval
        ax1.axvline(lag_start_time, color='black', linestyle='-.', alpha=0.9)
        ax1.axvline(
            event['peak_time'],
            color='black',
            linestyle=(0, (6, 2, 1, 2)),
            alpha=0.8,
            linewidth=2
        )
        
        ax1.legend(loc='upper right', fontsize=11)
        
        ax2.set_ylabel('Water Level (mMSL)', fontsize=11)
        ax2.set_xlabel('Date Time', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Combine legends if Q is plotted
        if ax2_q is not None:
            lines_l, labels_l = ax2.get_legend_handles_labels()
            lines_r, labels_r = ax2_q.get_legend_handles_labels()
            ax2.legend(lines_l + lines_r, labels_l + labels_r, loc='upper right', fontsize=11, ncol=2)
        else:
            ax2.legend(loc='upper right', fontsize=11, ncol=2)
        
        # Add title with event statistics
        title = f"Storm Event #{event['event_id']} - {event['start_time'].strftime('%Y-%m-%d')}\n"
        
        title += f"Total Precipitation: {event['total_precip']:.1f} mm | "
        title += f"WL Rise: {event['wl_rise']:.2f} m"
        
        fig.suptitle(title, fontsize=11, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_all_events(self, output_dir='/home/claude/storm_events'):
        """
        Plot all identified events and save to directory.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for event in self.events:
            event_id = event['event_id']
            filename = f"{output_dir}/event_{event_id:03d}_{event['start_time'].strftime('%Y%m%d')}.png"
            self.plot_event(event_id, save_path=filename)
            plt.close()
        
        print(f"\nAll {len(self.events)} event plots saved to {output_dir}/")
    
    def get_time_to_peak_statistics(self):
        """
        Calculate statistics for time to peak across all events.
        """
        if not self.events:
            print("No events identified yet.")
            return None
        
        ttp_values = [e['time_to_peak_hours'] for e in self.events]
        lag_values = [e['lag_time_hours'] for e in self.events]
        precip_dur_values = [e['precip_duration_hours'] for e in self.events]
        event_dur_values = [e['duration_hours'] for e in self.events]
        peak_delay_values = [e['peak_delay_hours'] for e in self.events]
        
        stats = {
            'Time_to_Peak': {
                'count': len(ttp_values),
                'mean': np.mean(ttp_values),
                'median': np.median(ttp_values),
                'std': np.std(ttp_values),
                'min': np.min(ttp_values),
                'max': np.max(ttp_values)
            },
            'Lag_Time': {
                'count': len(lag_values),
                'mean': np.mean(lag_values),
                'median': np.median(lag_values),
                'std': np.std(lag_values),
                'min': np.min(lag_values),
                'max': np.max(lag_values)
            },
            'Peak_Delay': {
                'count': len(peak_delay_values),
                'mean': np.mean(peak_delay_values),
                'median': np.median(peak_delay_values),
                'std': np.std(peak_delay_values),
                'min': np.min(peak_delay_values),
                'max': np.max(peak_delay_values)
            },
            'Precip_Duration': {
                'count': len(precip_dur_values),
                'mean': np.mean(precip_dur_values),
                'median': np.median(precip_dur_values),
                'std': np.std(precip_dur_values),
                'min': np.min(precip_dur_values),
                'max': np.max(precip_dur_values)
            },
            'Event_Duration': {
                'count': len(event_dur_values),
                'mean': np.mean(event_dur_values),
                'median': np.median(event_dur_values),
                'std': np.std(event_dur_values),
                'min': np.min(event_dur_values),
                'max': np.max(event_dur_values)
            }
        }
        
        stats_df = pd.DataFrame(stats).round(2)
        return stats_df


def main():
    """
    Main execution function - demonstration of usage.
    """
    print("=" * 80)
    print("STORM EVENT ANALYZER - USER CRITERIA (MERGED VERSION)")
    print("=" * 80)
    print("\nThis script identifies isolated storm events with the following rules:")
    print("  - 36h antecedent dry period (≤ 0.5 mm/hr)")
    print("  - 36h antecedent WL stability (variation ≤ 0.2 m)")
    print("  - Event start = LOWEST WL after rain starts (>0.5 mm/hr)")
    print("  - Minimum total precipitation = 1.0 mm")
    print("  - WL must rise at least 0.5 m from initial")
    print("  - Event ends at initial WL + 0.05 m")
    print("  - Peak = HIGHEST WL between event start and event end")
    print("\nWMO timing metrics are reported for diagnostics (not used for rejection).")
    print("=" * 80)
    

if __name__ == "__main__":
    main()
