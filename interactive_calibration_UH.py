"""
Interactive UH Parameter Tuning
Manually adjust parameters and see results immediately
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
from scipy.signal import convolve
from scipy.special import gamma as gamma_func
import os
import re
from datetime import datetime


def prepare_event_data_for_calibration(df, event, precip_col='corr_precip',
                                       baseflow_method='linear',
                                       baseflow_k=0.95, q_col='Q',
                                       phi_index=1.0):
    """Prepare effective rainfall and observed direct runoff for one event."""
    if df is None or event is None:
        raise ValueError('df and event must be provided')

    start_idx = int(event.get('start_idx'))
    end_idx = int(event.get('end_idx'))
    if end_idx <= start_idx:
        raise ValueError(f'Invalid event window ({start_idx} to {end_idx})')

    event_df = df.iloc[start_idx:end_idx + 1].copy()
    if event_df.empty:
        raise ValueError('Event dataframe is empty')

    if precip_col not in event_df.columns:
        raise KeyError(f'Column {precip_col} not found in event data')
    if q_col not in event_df.columns:
        raise KeyError(f'Column {q_col} not found in event data')

    precip = np.asarray(event_df[precip_col].astype(float).to_numpy(), dtype=float)
    q_total = np.asarray(event_df[q_col].astype(float).to_numpy(), dtype=float)

    if baseflow_method == 'constant':
        q_base = np.full(len(q_total), (q_total[0] + q_total[-1]) / 2, dtype=float)
    elif baseflow_method == 'linear':
        q_base = np.linspace(q_total[0], q_total[-1], len(q_total), dtype=float)
    elif baseflow_method == 'average':
        n = max(1, min(6, len(q_total) // 10))
        q_start_avg = q_total[:n].mean() if n > 0 else q_total[0]
        q_end_avg = q_total[-n:].mean() if n > 0 else q_total[-1]
        q_base = np.linspace(q_start_avg, q_end_avg, len(q_total), dtype=float)
    elif baseflow_method == 'recession':
        q_base = np.zeros_like(q_total, dtype=float)
        q_base[0] = q_total[0]
        for i in range(1, len(q_total)):
            q_base[i] = baseflow_k * q_base[i - 1]
            if q_base[i] > q_total[i]:
                q_base[i] = q_total[i]
    else:
        raise ValueError(f'Unknown baseflow method: {baseflow_method}')

    q_direct = q_total - q_base
    q_direct[q_direct < 0] = 0

    effective_rainfall = precip - float(phi_index)
    effective_rainfall[effective_rainfall < 0] = 0

    return {
        'effective_rainfall': effective_rainfall,
        'observed_direct_runoff': q_direct,
        'phi_index': float(phi_index),
        'event_id': event.get('event_id'),
        'event_info': event,
    }


class InteractiveUHCalibration:
    """
    Interactive tool to manually calibrate UH parameters
    """
    
    def __init__(self, event_data, watershed_area_km2, model_type='triangular',
                 df=None, event=None, precip_col='corr_precip', 
                 baseflow_method='linear', baseflow_k=0.95, q_col='Q'):
        """
        Parameters:
        -----------
        event_data : dict
            From prepare_event_data_for_calibration
        watershed_area_km2 : float
            Watershed area
        model_type : str
            'triangular' or 'nash'
        df : DataFrame (optional)
            Full dataset for recalculating with different phi_index
        event : dict (optional)
            Event dictionary for recalculating with different phi_index
        precip_col : str
            Precipitation column name
        baseflow_method : str
            Baseflow separation method
        baseflow_k : float
            Recession constant for baseflow
        q_col : str
            Discharge column name
        """
        self.event_data = event_data
        self.area = watershed_area_km2
        self.model_type = model_type
        
        # Store data for phi_index recalculation
        self.df = df
        self.event = event
        self.precip_col = precip_col
        self.baseflow_method = baseflow_method
        self.baseflow_k = baseflow_k
        self.q_col = q_col
        
        # Get observed data
        self.P_eff = event_data['effective_rainfall']
        self.Q_obs = event_data['observed_direct_runoff']
        self.t = np.arange(len(self.Q_obs))
        
        # Initial parameters
        if model_type == 'triangular':
            self.Tp = 10.0
            self.Tb = 50.0
        else:  # nash
            self.K = 5.0
            self.n = 3.0
        
        # Phi-index parameter
        self.phi_index = event_data.get('phi_index', 1.0)
        
        # Setup plot
        self.setup_plot()
    
    def nash_uh(self, t, K, n):
        """Nash/Gamma UH"""
        h = np.zeros_like(t, dtype=float)
        mask = t > 0
        h[mask] = (1 / (K * gamma_func(n))) * ((t[mask] / K) ** (n - 1)) * np.exp(-t[mask] / K)
        
        # Normalize
        if np.sum(h) > 0:
            h = h / (np.sum(h) * (t[1] - t[0]))
        
        return h
    
    def triangular_uh(self, t, Tp, Tb):
        """Triangular UH"""
        if Tp <= 0 or Tb <= Tp:
            return np.zeros_like(t)
        
        Qp = 2.0 / Tb
        h = np.zeros_like(t)
        
        # Rising limb
        rising = (t >= 0) & (t <= Tp)
        h[rising] = (Qp / Tp) * t[rising]
        
        # Recession limb
        recession = (t > Tp) & (t <= Tb)
        h[recession] = Qp - (Qp / (Tb - Tp)) * (t[recession] - Tp)
        
        return h
    
    def simulate_runoff(self, params):
        """Simulate runoff with given parameters"""
        t_uh = np.arange(0, len(self.P_eff) * 2, 1.0)
        
        # Generate UH
        if self.model_type == 'triangular':
            Tp, Tb = params
            uh = self.triangular_uh(t_uh, Tp, Tb)
        else:  # nash
            K, n = params
            uh = self.nash_uh(t_uh, K, n)
        
        # Convert to m³/s per mm
        uh_m3s_mm = uh * self.area * 1e3 / 3600
        
        # Convolve
        Q_sim = convolve(self.P_eff, uh_m3s_mm, mode='full')[:len(self.P_eff)]
        
        return Q_sim, uh
    
    def recalculate_effective_rainfall(self, phi_index):
        """Recalculate effective rainfall and direct runoff with new phi_index"""
        if self.df is None or self.event is None:
            return None
        
        # Extract event window
        start_idx = self.event['start_idx']
        end_idx = self.event['end_idx']
        event_df = self.df.iloc[start_idx:end_idx+1].copy()
        
        # Get data
        precip = event_df[self.precip_col].values
        Q_total = event_df[self.q_col].values
        
        # Recalculate baseflow
        if self.baseflow_method == 'constant':
            Q_base = np.full_like(Q_total, (Q_total[0] + Q_total[-1]) / 2)
        elif self.baseflow_method == 'linear':
            Q_base = np.linspace(Q_total[0], Q_total[-1], len(Q_total))
        elif self.baseflow_method == 'average':
            N = min(6, len(Q_total) // 10)
            Q_start_avg = Q_total[:N].mean() if N > 0 else Q_total[0]
            Q_end_avg = Q_total[-N:].mean() if N > 0 else Q_total[-1]
            Q_base = np.linspace(Q_start_avg, Q_end_avg, len(Q_total))
        elif self.baseflow_method == 'recession':
            Q_base = np.zeros_like(Q_total)
            Q_base[0] = Q_total[0]
            for i in range(1, len(Q_total)):
                Q_base[i] = self.baseflow_k * Q_base[i-1]
                if Q_base[i] > Q_total[i]:
                    Q_base[i] = Q_total[i]
        
        # Direct runoff
        Q_direct = Q_total - Q_base
        Q_direct[Q_direct < 0] = 0
        
        # Effective rainfall
        effective_rainfall = precip - phi_index
        effective_rainfall[effective_rainfall < 0] = 0
        
        return effective_rainfall, Q_direct
    
    def calculate_nse(self, Q_obs, Q_sim):
        """Calculate NSE"""
        if len(Q_obs) == 0 or np.std(Q_obs) == 0:
            return 0
        
        numerator = np.sum((Q_obs - Q_sim) ** 2)
        denominator = np.sum((Q_obs - np.mean(Q_obs)) ** 2)
        
        if denominator == 0:
            return 0
        
        nse = 1 - (numerator / denominator)
        return nse
    
    def setup_plot(self):
        """Setup interactive plot"""
        self.fig = plt.figure(figsize=(12, 8))
        
        # Create grid
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.3)
        
        # Axes
        self.ax_rain = self.fig.add_subplot(gs[0, 0])
        self.ax_compare = self.fig.add_subplot(gs[1, 0])
        
        # Initial simulation
        if self.model_type == 'triangular':
            params = (self.Tp, self.Tb)
        else:
            params = (self.K, self.n)
        
        Q_sim, uh = self.simulate_runoff(params)
        nse = self.calculate_nse(self.Q_obs, Q_sim)
        
        # Plot rainfall
        self.ax_rain.clear()
        self.ax_rain.bar(self.t, self.P_eff, color='0.15', alpha=0.8,
                         label='Effective rainfall')
        self.ax_rain.set_ylabel('Rainfall (mm/hr)', fontsize=11)
        self.ax_rain.set_title('Effective Rainfall', fontsize=12, fontweight='bold')
        self.ax_rain.invert_yaxis()
        self.ax_rain.grid(True, alpha=0.3)
        self.ax_rain.legend(fontsize=10)
        
        # Plot comparison
        self.line_obs, = self.ax_compare.plot(
            self.t, self.Q_obs, color='0.25', linewidth=2.2,
            marker='o', markersize=4, linestyle='-', label='Observed', alpha=0.9
        )
        self.line_sim, = self.ax_compare.plot(
            self.t, Q_sim, color='0.55', linewidth=2.2,
            marker='s', markersize=4, linestyle='--', label='Simulated', alpha=0.9
        )
        self.ax_compare.set_xlabel('Time (hours)', fontsize=11)
        self.ax_compare.set_ylabel('Direct Runoff (m³/s)', fontsize=11)
        self.ax_compare.legend(fontsize=11)
        self.ax_compare.grid(True, alpha=0.3)
        
        # Title with event metadata and NSE
        start_dt = self.event.get('start_time') or self.event.get('start_datetime')
        if start_dt is None:
            start_dt = self.event.get('start_idx')

        start_label = pd.Timestamp(start_dt).strftime('%Y-%m-%d %H:%M') if pd.notna(start_dt) and not isinstance(start_dt, (int, float)) else str(start_dt)
        event_id = self.event.get('event_id', self.event_data.get('event_id'))

        self.title_text = self.ax_compare.set_title(
            f'Observed vs Simulated\nEvent ID: {event_id} | Start: {start_label} | NSE = {nse:.3f}',
            fontsize=12, fontweight='bold'
        )
        
        # Create sliders
        plt.subplots_adjust(bottom=0.30)
        
        if self.model_type == 'triangular':
            # Tp slider
            ax_tp = plt.axes((0.15, 0.18, 0.7, 0.03))
            self.slider_tp = Slider(ax_tp, 'Tp (hours)', 1.0, 50.0, 
                                   valinit=self.Tp, valstep=0.5)
            self.slider_tp.on_changed(self.update)
            
            # Tb slider
            ax_tb = plt.axes((0.15, 0.13, 0.7, 0.03))
            self.slider_tb = Slider(ax_tb, 'Tb (hours)', 10.0, 200.0, 
                                   valinit=self.Tb, valstep=1.0)
            self.slider_tb.on_changed(self.update)
            
        else:  # nash
            # K slider
            ax_k = plt.axes((0.15, 0.18, 0.7, 0.03))
            self.slider_k = Slider(ax_k, 'K (hours)', 0.5, 30.0, 
                                  valinit=self.K, valstep=0.5)
            self.slider_k.on_changed(self.update)
            
            # n slider
            ax_n = plt.axes((0.15, 0.13, 0.7, 0.03))
            self.slider_n = Slider(ax_n, 'n', 0.5, 15.0, 
                                  valinit=self.n, valstep=0.5)
            self.slider_n.on_changed(self.update)
        
        # Phi-index slider (for all models)
        if self.df is not None and self.event is not None:
            precip = self.df.iloc[self.event['start_idx']:self.event['end_idx']+1][self.precip_col].values
            max_phi = precip.max() * 1.5  # Allow up to 1.5x max hourly rainfall
            
            ax_phi = plt.axes((0.15, 0.08, 0.7, 0.03))
            self.slider_phi = Slider(ax_phi, 'φ-index (mm/hr)', 0.0, max_phi, 
                                    valinit=self.phi_index, valstep=0.1)
            self.slider_phi.on_changed(self.update)
    
    def update(self, val):
        """Update plot when sliders change"""
        # Get phi-index if slider exists
        if hasattr(self, 'slider_phi'):
            phi_index = self.slider_phi.val
            # Recalculate effective rainfall if data available
            if self.df is not None and self.event is not None:
                result = self.recalculate_effective_rainfall(phi_index)
                if result:
                    P_eff, Q_direct = result
                    self.P_eff = P_eff
                    self.Q_obs = Q_direct
                    self.phi_index = phi_index
        else:
            phi_index = self.phi_index
        
        # Get parameters
        if self.model_type == 'triangular':
            Tp = self.slider_tp.val
            Tb = self.slider_tb.val
            params = (Tp, Tb)
            param_text = f'Tp={Tp:.1f}h, Tb={Tb:.1f}h'
        else:
            K = self.slider_k.val
            n = self.slider_n.val
            params = (K, n)
            param_text = f'K={K:.1f}h, n={n:.1f}'
        
        # Add phi-index to param text
        param_text += f', φ={phi_index:.2f}mm/hr'
        
        # Simulate
        Q_sim, uh = self.simulate_runoff(params)
        
        # Calculate NSE
        nse = self.calculate_nse(self.Q_obs, Q_sim)
        
        # Update rainfall plot (in case phi changed)
        self.ax_rain.clear()
        self.ax_rain.bar(self.t, self.P_eff, color='0.15', alpha=0.8,
                         label='Effective rainfall')
        self.ax_rain.set_ylabel('Rainfall (mm/hr)', fontsize=11)
        self.ax_rain.set_title('Effective Rainfall', fontsize=12, fontweight='bold')
        self.ax_rain.invert_yaxis()
        self.ax_rain.grid(True, alpha=0.3)
        self.ax_rain.legend(fontsize=10)
        
        # Update comparison
        self.line_sim.set_ydata(Q_sim)
        self.ax_compare.relim()
        self.ax_compare.autoscale_view()
        
        # Update title
        start_dt = self.event.get('start_time') or self.event.get('start_datetime')
        if start_dt is None:
            start_dt = self.event.get('start_idx')
        start_label = pd.Timestamp(start_dt).strftime('%Y-%m-%d %H:%M') if pd.notna(start_dt) and not isinstance(start_dt, (int, float)) else str(start_dt)
        event_id = self.event.get('event_id', self.event_data.get('event_id'))
        self.title_text.set_text(
            f'Observed vs Simulated\nEvent ID: {event_id} | Start: {start_label} | NSE = {nse:.3f} | {param_text}'
        )
        
        # Color code NSE
        if nse > 0.7:
            self.title_text.set_color('green')
        elif nse > 0.5:
            self.title_text.set_color('orange')
        else:
            self.title_text.set_color('red')
        
        self.fig.canvas.draw_idle()
    
    def show(self, save_path=None):
        """Display the interactive plot and optionally save when closed"""
        if save_path:
            def _save_on_close(event):
                # Export a cleaner image: hide slider controls and phi value in title.
                original_title = self.title_text.get_text()
                original_subplotpars = (
                    self.fig.subplotpars.left,
                    self.fig.subplotpars.right,
                    self.fig.subplotpars.bottom,
                    self.fig.subplotpars.top,
                    self.fig.subplotpars.wspace,
                    self.fig.subplotpars.hspace,
                )

                sliders_to_hide = []
                if self.model_type == 'triangular':
                    if hasattr(self, 'slider_tp'):
                        sliders_to_hide.append(self.slider_tp)
                    if hasattr(self, 'slider_tb'):
                        sliders_to_hide.append(self.slider_tb)
                else:
                    if hasattr(self, 'slider_k'):
                        sliders_to_hide.append(self.slider_k)
                    if hasattr(self, 'slider_n'):
                        sliders_to_hide.append(self.slider_n)
                if hasattr(self, 'slider_phi'):
                    sliders_to_hide.append(self.slider_phi)

                for slider in sliders_to_hide:
                    slider.ax.set_visible(False)

                cleaned_title = re.sub(r",\s*φ=[^\n|]*mm/hr", "", original_title)
                self.title_text.set_text(cleaned_title)

                # Reduce extra bottom space left by slider area before exporting.
                self.fig.subplots_adjust(bottom=0.12)
                self.fig.canvas.draw_idle()
                self.fig.savefig(save_path, dpi=200, bbox_inches='tight')

                # Restore interactive view state.
                self.title_text.set_text(original_title)
                for slider in sliders_to_hide:
                    slider.ax.set_visible(True)
                self.fig.subplots_adjust(
                    left=original_subplotpars[0],
                    right=original_subplotpars[1],
                    bottom=original_subplotpars[2],
                    top=original_subplotpars[3],
                    wspace=original_subplotpars[4],
                    hspace=original_subplotpars[5],
                )
            self.fig.canvas.mpl_connect('close_event', _save_on_close)
        plt.show()
    
    def get_current_params(self):
        """Get current parameter values"""
        params = {}
        
        if self.model_type == 'triangular':
            params['Tp'] = self.slider_tp.val
            params['Tb'] = self.slider_tb.val
        else:
            params['K'] = self.slider_k.val
            params['n'] = self.slider_n.val
        
        if hasattr(self, 'slider_phi'):
            params['phi_index'] = self.slider_phi.val
        else:
            params['phi_index'] = self.phi_index
        
        return params


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import pandas as pd

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'PREMONSOON-ALL-FILLEDv5_corrected.xlsx')
    summary_path = os.path.join(base_dir, 'identified_events', 'version-5imp', 'storm_events_summary.xlsx')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Input data not found: {data_path}')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f'Storm summary not found: {summary_path}')
    
    # Load data
    df = pd.read_excel(data_path)
    df['Date Time'] = pd.to_datetime(df['Date Time'])
    summary = pd.read_excel(summary_path)
    summary["Start_Time"] = pd.to_datetime(summary["Start_Time"])
    summary["End_Time"] = pd.to_datetime(summary["End_Time"])
    summary.head()
    
    # Build event dictionaries required by derive_uh_from_storm_event()
    events = []
    for i, (index, row) in enumerate(summary.iterrows()):
        event_id = int(row.get("Event_ID", i + 1))
        start_time = row["Start_Time"]
        end_time = row["End_Time"]
    
        start_match = df.index[df["Date Time"] == start_time]
        end_match = df.index[df["Date Time"] == end_time]
    
        if len(start_match) > 0:
            start_idx = int(start_match[0])
        else:
            start_idx = int((df["Date Time"] - start_time).abs().idxmin())
    
        if len(end_match) > 0:
            end_idx = int(end_match[0])
        else:
            end_idx = int((df["Date Time"] - end_time).abs().idxmin())
    
        if end_idx <= start_idx:
            print(f"Skipping Event {event_id}: invalid window ({start_idx} to {end_idx})")
            continue
    
        events.append(
            {
                "event_id": event_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": start_time,
                "end_time": end_time,
            }
        )
    
    print(f"Prepared {len(events)} events from summary dataframe")

    # Setup
    WATERSHED_AREA = float(input("Enter watershed area (km²): "))
    output_dir = os.path.join(base_dir, "interactive_calibration_plots")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("INTERACTIVE UH CALIBRATION")
    print("="*80)
    print(f"Total detected events: {len(events)}")

    # Choose model
    model = input("\nModel type (triangular/nash): ").strip().lower()
    if model not in ['triangular', 'nash']:
        model = 'triangular'

    # Choose event scope
    scope = input("Calibrate event index (0-based) or 'all' [all]: ").strip().lower()
    if scope == "":
        scope = "all"

    if scope == 'all':
        selected_indices = list(range(len(events)))
    else:
        try:
            idx = int(scope)
            if idx < 0 or idx >= len(events):
                raise ValueError
            selected_indices = [idx]
        except ValueError:
            print("Invalid selection. Falling back to all events.")
            selected_indices = list(range(len(events)))

    print("\nInstructions:")
    print("  1. Adjust sliders to change UH parameters and phi-index")
    print("  2. Watch how simulated Q changes")
    print("  3. Try to match observed Q (blue line)")
    print("  4. NSE turns GREEN when > 0.7")
    print("  5. Close window to move to next event")

    all_results = []

    for idx in selected_indices:
        event = events[idx]
        event['watershed_area'] = WATERSHED_AREA

        # Resolve event start/end timestamps for CSV output.
        start_dt = event.get('start_time') or event.get('start_datetime')
        end_dt = event.get('end_time') or event.get('end_datetime')
        if start_dt is None:
            start_dt = df.iloc[event['start_idx']]['Date Time']
        if end_dt is None:
            end_dt = df.iloc[event['end_idx']]['Date Time']

        event_data = prepare_event_data_for_calibration(
            df, event,
            baseflow_method='linear',
            phi_index=1.0
        )

        print("\n" + "-"*80)
        print(f"Event list index: {idx} | Event ID: {event['event_id']}")
        print(f"Effective rainfall: {event_data['effective_rainfall'].sum():.1f} mm")
        print(f"Observed peak Q: {event_data['observed_direct_runoff'].max():.2f} m³/s")

        tool = InteractiveUHCalibration(
            event_data,
            WATERSHED_AREA,
            model_type=model,
            df=df,
            event=event,
            precip_col='corr_precip',
            baseflow_method='linear',
            q_col='Q'
        )
        plot_filename = (
            f"event_idx{idx:03d}_id{event['event_id']}_{model}.png"
        )
        plot_path = os.path.join(output_dir, plot_filename)
        tool.show(save_path=plot_path)

        final_params = tool.get_current_params()
        all_results.append({
            'event_list_index': idx,
            'event_id': event['event_id'],
            'event_start': pd.to_datetime(start_dt),
            'event_end': pd.to_datetime(end_dt),
            **final_params
        })

        print("\nFinal parameters for this event:")
        for key, value in final_params.items():
            print(f"  {key} = {value:.2f}")
        print(f"Saved plot: {plot_path}")

    print("\n" + "="*80)
    print("CALIBRATION SUMMARY")
    print("="*80)
    for res in all_results:
        params_txt = ", ".join(
            f"{k}={v:.2f}" for k, v in res.items() if k not in ['event_list_index', 'event_id']
        )
        print(f"Index {res['event_list_index']} | Event ID {res['event_id']} -> {params_txt}")

    # Export calibration summary to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"calibration_summary_{model}_{timestamp}.csv")
    pd.DataFrame(all_results).to_csv(csv_path, index=False)
    print(f"\nSaved CSV summary: {csv_path}")