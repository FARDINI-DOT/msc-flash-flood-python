"""
Seasonal Unit Hydrograph Discharge Simulation
==============================================
Simulates discharge using calibrated UH parameters for:
1. Pre-monsoon (March-May)
2. Monsoon (June-September)
3. Full timeseries comparison

Uses final calibrated parameters:
  Tp = 15.1 hours
  Tb = 81.5 hours
  Average phi-index by season
"""

import pandas as pd
import numpy as np
from scipy.signal import convolve
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


class SeasonalUHSimulator:
    """
    Simulate discharge using triangular unit hydrograph with seasonal parameters
    """
    
    def __init__(self, watershed_area_km2):
        """
        Initialize with watershed area
        
        Parameters:
        -----------
        watershed_area_km2 : float
            Watershed area in km²
        """
        self.area = watershed_area_km2
        
        # FINAL CALIBRATED PARAMETERS
        self.Tp = 27  # hours
        self.Tb = 124  # hours
        
        # SEASONAL PHI-INDEX (from your calibration)
        self.phi_monsoon = 0.5  # mm/hr (June-September, saturated)
        self.phi_premonsoon = 2  # mm/hr (March-May, dry)
        self.phi_postmonsoon = 1.5  # mm/hr (October-November)
        self.phi_winter = 1.5  # mm/hr (December-February)
        
        print("="*70)
        print("SEASONAL UNIT HYDROGRAPH SIMULATOR")
        print("="*70)
        print(f"Watershed Area: {self.area:.2f} km²")
        print(f"\nCalibrated UH Parameters:")
        print(f"  Time to Peak (Tp): {self.Tp:.1f} hours")
        print(f"  Time Base (Tb): {self.Tb:.1f} hours")
        print(f"  Tb/Tp ratio: {self.Tb/self.Tp:.2f}")
        print(f"\nSeasonal Phi-Index (mm/hr):")
        print(f"  Monsoon (Jun-Sep):     {self.phi_monsoon:.2f}")
        print(f"  Pre-monsoon (Mar-May): {self.phi_premonsoon:.2f}")
        print(f"  Post-monsoon (Oct-Nov): {self.phi_postmonsoon:.2f}")
        print(f"  Winter (Dec-Feb):      {self.phi_winter:.2f}")
        print("="*70)
    
    def get_seasonal_phi(self, month):
        """
        Get phi-index based on month
        
        Parameters:
        -----------
        month : int
            Month number (1-12)
            
        Returns:
        --------
        phi : float
            Phi-index in mm/hr
        """
        if month in [6, 7, 8, 9]:  # Monsoon
            return self.phi_monsoon
        elif month in [3, 4, 5]:  # Pre-monsoon
            return self.phi_premonsoon
        elif month in [10, 11]:  # Post-monsoon
            return self.phi_postmonsoon
        else:  # Winter
            return self.phi_winter

    def get_season_months(self, season):
        """
        Return the month numbers for a named season
        """
        season_months = {
            'monsoon': [6, 7, 8, 9],
            'premonsoon': [3, 4, 5],
            'postmonsoon': [10, 11],
            'winter': [12, 1, 2]
        }
        return season_months.get(season)

    def filter_by_season(self, df, season):
        """
        Filter a dataframe to a named season and return a plot label
        """
        season_labels = {
            'monsoon': 'Monsoon (June-September)',
            'premonsoon': 'Pre-monsoon (March-May)',
            'postmonsoon': 'Post-monsoon (October-November)',
            'winter': 'Winter (December-February)',
            'all': 'Full Timeseries'
        }
        months = self.get_season_months(season)
        if months is None:
            return df, season_labels['all']
        return df[df['Date Time'].dt.month.isin(months)], season_labels[season]

    def find_water_level_column(self, df):
        """
        Find a likely water level column in the input dataframe.
        """
        candidates = [
            'wl mmsl',
            'wl (mmsl)',
            'wl(mmsl)',
            'water_level',
            'water level',
            'wl',
            'stage',
            'h',
            'level'
        ]
        normalized = {c.lower().strip(): c for c in df.columns}

        for name in candidates:
            if name in normalized:
                return normalized[name]

        # Fallback: normalized partial match for common water level keywords
        for col in df.columns:
            low = col.lower()
            low_compact = low.replace(' ', '').replace('_', '').replace('(', '').replace(')', '')
            if ('water' in low and 'level' in low) or ('stage' in low) or (low.startswith('wl')):
                return col
            if 'wlmmsl' in low_compact:
                return col

        return None

    def calculate_performance_metrics(self, observed, simulated):
        """
        Calculate performance metrics for paired observed and simulated series
        """
        observed = np.asarray(observed)
        simulated = np.asarray(simulated)

        residuals = observed - simulated
        denominator = np.sum((observed - observed.mean())**2)
        nse = 1 - (np.sum(residuals**2) / denominator) if denominator > 0 else -999
        rmse = np.sqrt(np.mean(residuals**2))
        pbias = 100 * np.sum(residuals) / np.sum(observed) if np.sum(observed) > 0 else 0

        spearman = np.nan
        if len(observed) > 1 and np.std(observed) > 0 and np.std(simulated) > 0:
            spearman = spearmanr(observed, simulated).statistic

        return {
            'residuals': residuals,
            'NSE': nse,
            'RMSE': rmse,
            'PBIAS': pbias,
            'Spearman': spearman
        }
    
    def triangular_uh(self, t, Tp, Tb):
        """
        Generate triangular unit hydrograph ordinates
        
        Parameters:
        -----------
        t : array
            Time array (hours)
        Tp : float
            Time to peak (hours)
        Tb : float
            Time base (hours)
            
        Returns:
        --------
        h : array
            UH ordinates (1/hour)
        """
        Qp = 2.0 / Tb
        h = np.zeros_like(t, dtype=float)
        
        # Rising limb
        rising = (t >= 0) & (t <= Tp)
        h[rising] = (Qp / Tp) * t[rising]
        
        # Recession limb
        recession = (t > Tp) & (t <= Tb)
        h[recession] = Qp - (Qp / (Tb - Tp)) * (t[recession] - Tp)
        # Everything beyond Tb is zero
        beyond_base = t > Tb
        h[beyond_base] = 0.0

        # Safety clip
        h = np.clip(h, 0, None)
        return h
    
    def calculate_effective_rainfall(self, df, precip_col='corr_precip'):
        """
        Calculate effective rainfall using seasonal phi-index
        
        Parameters:
        -----------
        df : DataFrame
            Input data with datetime and precipitation
        precip_col : str
            Precipitation column name
            
        Returns:
        --------
        df : DataFrame
            DataFrame with effective rainfall column added
        """
        df = df.copy()
        
        # Get phi for each timestep based on month
        df['phi_index'] = df['Date Time'].dt.month.apply(self.get_seasonal_phi)
        
        # Calculate effective rainfall (mm/hr)
        df['P_effective'] = df[precip_col] - df['phi_index']
        df['P_effective'] = df['P_effective'].clip(lower=0)  # No negative
        
        return df
    
    def simulate_discharge(self, df, precip_col='corr_precip'):
        """
        Simulate discharge using unit hydrograph convolution
        Uses Q_direct from input file (no baseflow calculation needed)
        
        Parameters:
        -----------
        df : DataFrame
            Input data with datetime and precipitation
        precip_col : str
            Precipitation column name
            
        Returns:
        --------
        df : DataFrame
            DataFrame with simulated discharge added
        """
        print("\nSimulating discharge...")
        print("  Using Q_direct from input file (no baseflow calculation)")
        
        # Calculate effective rainfall
        df = self.calculate_effective_rainfall(df, precip_col)
        
        # Generate unit hydrograph
        max_time = int(self.Tb * 3)  # Extended time for complete response
        t_uh = np.arange(0, max_time, 1.0)
        uh = self.triangular_uh(t_uh, self.Tp, self.Tb)
        
        # Convert UH to discharge units (m³/s per mm of rain)
        # UH (1/hr) * Area (km²) * 1000 (mm to m³/km²) / 3600 (hr to sec)
        uh_m3s_mm = uh * self.area * 1000 / 3600
        
        # Convolve effective rainfall with UH
        P_eff = df['P_effective'].values
        # FIX 1: Use mode='same' (center of convolution, no edge artifacts)
        Q_direct_simulated = convolve(P_eff, uh_m3s_mm, mode='full')[:len(P_eff)]

        # Convolve (full length)
        Q_direct_simulated = convolve(P_eff, uh_m3s_mm, mode='full')
        Q_direct_simulated = np.clip(Q_direct_simulated, 0, None)

        # Extend dataframe to match
        n_extra = len(Q_direct_simulated) - len(df)
        if n_extra > 0:
            last_date = df['Date Time'].iloc[-1]
            new_dates = [last_date + pd.Timedelta(hours=i+1) for i in range(n_extra)]
            df = pd.concat([df, pd.DataFrame({'Date Time': new_dates})], ignore_index=True)

        # Now assign
        df['Q_direct_simulated'] = Q_direct_simulated

# FIX 2: Clip to zero (physics constraint - discharge can't be negative)
        #Q_direct_simulated = np.clip(Q_direct_simulated, 0, None)
        
        # Use Q_direct from input file
        if 'Q_direct' in df.columns:
            print("  Found Q_direct in input file")
            df['Q_direct_observed'] = df['Q_direct']
            Q_direct_obs = df['Q_direct'].values
            
            # If both Q and Q_direct are available, calculate baseflow
            if 'Q' in df.columns:
                df['Q_baseflow_observed'] = df['Q'] - df['Q_direct']
                df['Q_baseflow_observed'] = df['Q_baseflow_observed'].clip(lower=0)  # No negative
                BFI = df['Q_baseflow_observed'].mean() / df['Q'].mean()
                print(f"  Baseflow Index (BFI): {BFI:.3f}")
                df['BFI'] = BFI
            else:
                df['Q_baseflow_observed'] = 0
                df['BFI'] = 0
        else:
            print("  WARNING: No Q_direct column found in input file!")
            print("  Make sure 'Q_direct' column exists in the Excel file")
            return df, {}
        
        # Store simulated components

        # Total simulated discharge combines simulated direct runoff with observed baseflow
        df['Q_simulated_total'] = Q_direct_simulated + df['Q_baseflow_observed']
        df['Q_simulated'] = df['Q_simulated_total']

        # Compare simulated direct runoff against observed direct runoff
        direct_metrics = self.calculate_performance_metrics(Q_direct_obs, df['Q_direct_simulated'].values)
        df['residual'] = direct_metrics['residuals']
        
        # Also calculate NSE on total discharge for reference (if Q is available)
        nse_total = -999
        if 'Q' in df.columns:
            total_metrics = self.calculate_performance_metrics(df['Q'].values, df['Q_simulated_total'].values)
            nse_total = total_metrics['NSE']
        
        print(f"\n  DIRECT RUNOFF Performance (UH validation):")
        print(f"    NSE (Q_direct):      {direct_metrics['NSE']:.3f}")
        print(f"    RMSE (Q_direct):     {direct_metrics['RMSE']:.2f} m³/s")
        print(f"    PBIAS (Q_direct):    {direct_metrics['PBIAS']:.1f}%")
        if pd.notna(direct_metrics['Spearman']):
            print(f"    Spearman (Q_direct): {direct_metrics['Spearman']:.3f}")
        if nse_total > -999:
            print(f"\n  TOTAL DISCHARGE Performance (reference only):")
            print(f"    NSE (Q_total):   {nse_total:.3f}")
        
        return df, {
            'NSE_direct': direct_metrics['NSE'], 
            'RMSE_direct': direct_metrics['RMSE'], 
            'PBIAS_direct': direct_metrics['PBIAS'],
            'Spearman_direct': direct_metrics['Spearman'],
            'NSE_total': nse_total
        }
    
    def plot_seasonal_comparison(self, df, season='monsoon', year=None, save_path=None):
        """
        Plot observed vs simulated discharge for a specific season
        
        Parameters:
        -----------
        df : DataFrame
            Data with observed and simulated discharge
        season : str
            'monsoon' (Jun-Sep), 'premonsoon' (Mar-May), or 'all'
        year : int
            Specific year to plot (optional)
        save_path : str
            Path to save figure
        """
        df_plot = df.copy()
        
        df_plot, season_name = self.filter_by_season(df_plot, season)
        
        # Filter by year if specified
        if year:
            df_plot = df_plot[df_plot['Date Time'].dt.year == year]
            season_name += f' ({year})'
        
        if len(df_plot) == 0:
            print(f"No data for {season_name}")
            return
        
        # Calculate statistics for this period
        if 'Q' in df_plot.columns and 'Q_direct_observed' in df_plot.columns:
            direct_metrics = self.calculate_performance_metrics(
                df_plot['Q_direct_observed'].values,
                df_plot['Q_direct_simulated'].values
            )
        else:
            direct_metrics = None
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Precipitation
        ax = axes[0]
        ax.bar(df_plot['Date Time'], df_plot['corr_precip'], 
             width=0.04, color='#0b3d91', alpha=0.9, label='Total Precipitation')
        ax.bar(df_plot['Date Time'], df_plot['P_effective'], 
             width=0.04, color='#001f5b', alpha=1.0, label='Effective Rainfall')
        ax.set_ylabel('Precipitation (mm/hr)', fontsize=11)
        ax.invert_yaxis()
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Discharge Simulation - {season_name}', fontsize=13, fontweight='bold')
        
        # Discharge comparison - DIRECT RUNOFF
        ax = axes[1]
        if 'Q_direct_observed' in df_plot.columns:
            ax.plot(df_plot['Date Time'], df_plot['Q_direct_observed'], 
                   'b-', linewidth=1.5, label='Observed Direct Runoff', alpha=0.7)
            ax.plot(df_plot['Date Time'], df_plot['Q_direct_simulated'], 
                   'r--', linewidth=1.5, label='Simulated Direct Runoff (UH)', alpha=0.8)
            ax.set_ylabel('Direct Runoff (m³/s)', fontsize=11)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if direct_metrics is not None:
                title_text = (
                    f"Direct Runoff: NSE = {direct_metrics['NSE']:.3f} | "
                    f"RMSE = {direct_metrics['RMSE']:.2f} m³/s"
                )
                if pd.notna(direct_metrics['Spearman']):
                    title_text += f" | Spearman = {direct_metrics['Spearman']:.3f}"
                ax.set_title(title_text, fontsize=11, fontweight='bold', color='darkblue')
        
        # Total discharge comparison (for reference)
        if 'Q' in df_plot.columns and 'Q_simulated_total' in df_plot.columns:
            ax_twin = ax.twinx()
            ax_twin.plot(df_plot['Date Time'], df_plot['Q'], 
                        'gray', linewidth=1, label='Total Q (obs)', alpha=0.3, linestyle='-')
            ax_twin.plot(df_plot['Date Time'], df_plot['Q_simulated_total'], 
                        'pink', linewidth=1, label='Total Q (sim)', alpha=0.3, linestyle='--')
            ax_twin.set_ylabel('Total Q (reference)', fontsize=9, color='gray')
            ax_twin.tick_params(axis='y', labelcolor='gray', labelsize=8)
            ax_twin.legend(loc='upper left', fontsize=8)
        
        # Residuals
        ax = axes[2]
        if 'Q' in df_plot.columns:
            ax.plot(df_plot['Date Time'], df_plot['residual'], 
                   'g-', linewidth=1, alpha=0.6)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
            ax.fill_between(df_plot['Date Time'], df_plot['residual'], 0, 
                           alpha=0.2, color='green')
        ax.set_ylabel('Residual (m³/s)', fontsize=11)
        ax.set_xlabel('Date', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig
    
    def plot_scatter(self, df, save_path=None):
        """
        Create scatter plot comparing direct runoff (the correct UH validation)
        """
        if 'Q_direct_observed' not in df.columns:
            print("No direct runoff data for scatter plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot 1: DIRECT RUNOFF (main validation)
        ax = axes[0]
        ax.scatter(df['Q_direct_observed'], df['Q_direct_simulated'], 
                  alpha=0.4, s=10, color='steelblue')
        
        max_val = max(df['Q_direct_observed'].max(), df['Q_direct_simulated'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 Line')
        
        # Statistics on DIRECT runoff
        direct_metrics = self.calculate_performance_metrics(
            df['Q_direct_observed'].values,
            df['Q_direct_simulated'].values
        )
        r2 = np.corrcoef(df['Q_direct_observed'], df['Q_direct_simulated'])[0, 1]**2
        
        ax.set_xlabel('Observed Direct Runoff (m³/s)', fontsize=11)
        ax.set_ylabel('Simulated Direct Runoff (m³/s)', fontsize=11)
        title_text = f"DIRECT RUNOFF Comparison (UH Validation)\nNSE = {direct_metrics['NSE']:.3f} | R² = {r2:.3f}"
        if pd.notna(direct_metrics['Spearman']):
            title_text += f" | Spearman = {direct_metrics['Spearman']:.3f}"
        ax.set_title(title_text, fontsize=11, fontweight='bold', color='darkblue')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Scatter plot 2: TOTAL DISCHARGE (reference only)
        if 'Q' in df.columns and 'Q_simulated_total' in df.columns:
            ax = axes[1]
            ax.scatter(df['Q'], df['Q_simulated_total'], 
                      alpha=0.4, s=10, color='gray')
            
            max_val = max(df['Q'].max(), df['Q_simulated_total'].max())
            ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 Line')
            
            # Statistics on TOTAL
            total_metrics = self.calculate_performance_metrics(df['Q'].values, df['Q_simulated_total'].values)
            r2_total = np.corrcoef(df['Q'], df['Q_simulated_total'])[0, 1]**2
            
            ax.set_xlabel('Observed Total Q (m³/s)', fontsize=11)
            ax.set_ylabel('Simulated Total Q (m³/s)', fontsize=11)
            ax.set_title(f"TOTAL Q Comparison (Reference)\nNSE = {total_metrics['NSE']:.3f} | R² = {r2_total:.3f}", 
                        fontsize=11, fontweight='bold', color='gray')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        
        return fig
    
    def plot_yearly_performance(self, df, year, save_path=None):
        """
        Plot full year performance with all components (interactive)
        
        Parameters:
        -----------
        df : DataFrame
            Data with simulations
        year : int
            Year to plot
        save_path : str
            Path to save figure
        """
        df_year = df[df['Date Time'].dt.year == year].copy()
        
        if len(df_year) == 0:
            print(f"  No data for {year}")
            return
        
        # Calculate statistics for this year (total discharge only)
        nse, rmse = None, None
        if 'Q' in df_year.columns and 'Q_simulated_total' in df_year.columns:
            total_metrics_year = self.calculate_performance_metrics(
                df_year['Q'].values,
                df_year['Q_simulated_total'].values
            )
            nse = total_metrics_year['NSE']
            rmse = total_metrics_year['RMSE']
        
        # Create interactive figure with 4 stacked subplots
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=(
                'Precipitation',
                'Total Discharge (Direct Runoff + Baseflow)',
                'Baseflow Separation',
                'Direct Runoff Only (UH Performance)'
            )
        )

        # 1. Precipitation (negative values for inverted visual style)
        fig.add_trace(
            go.Bar(
                x=df_year['Date Time'],
                y=-df_year['corr_precip'],
                name='Total Precip',
                marker_color='#0b3d91',
                opacity=0.95
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            go.Bar(
                x=df_year['Date Time'],
                y=-df_year['P_effective'],
                name='Effective Rainfall',
                marker_color='#001f5b',
                opacity=1.0
            ),
            row=1,
            col=1
        )

        # 2. Total discharge comparison
        if 'Q' in df_year.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_year['Date Time'],
                    y=df_year['Q'],
                    mode='lines',
                    name='Observed Total Q',
                    line=dict(color='#000000', width=2.2),
                    opacity=1.0
                ),
                row=2,
                col=1
            )
        fig.add_trace(
            go.Scatter(
                x=df_year['Date Time'],
                y=df_year['Q_simulated_total'],
                mode='lines',
                name='Simulated Total Q',
                line=dict(color='#4a4a4a', width=2.0, dash='dash'),
                opacity=1.0
            ),
            row=2,
            col=1
        )

        # 3. Baseflow separation
        if 'Q' in df_year.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_year['Date Time'],
                    y=df_year['Q'],
                    mode='lines',
                    name='Total Observed Q',
                    line=dict(color='#000000', width=2.0),
                    opacity=1.0
                ),
                row=3,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df_year['Date Time'],
                    y=df_year['Q_baseflow_observed'],
                    mode='lines',
                    name='Baseflow',
                    line=dict(color='#555555', width=2.0),
                    fill='tozeroy',
                    fillcolor='rgba(140, 140, 140, 0.18)',
                    opacity=1.0
                ),
                row=3,
                col=1
            )

        # 4. Direct runoff comparison
        if 'Q' in df_year.columns:
            Q_direct_obs = df_year['Q'] - df_year['Q_baseflow_observed']
            fig.add_trace(
                go.Scatter(
                    x=df_year['Date Time'],
                    y=Q_direct_obs,
                    mode='lines',
                    name='Observed Direct Runoff',
                    line=dict(color='#000000', width=2.2),
                    opacity=1.0
                ),
                row=4,
                col=1
            )
        fig.add_trace(
            go.Scatter(
                x=df_year['Date Time'],
                y=df_year['Q_direct_simulated'],
                mode='lines',
                name='Simulated Direct Runoff (UH)',
                line=dict(color='#4a4a4a', width=2.0, dash='dash'),
                opacity=1.0
            ),
            row=4,
            col=1
        )

        title = f'Unit Hydrograph Performance - Year {year} (Total Discharge)'
        if nse is not None:
            title += f' | NSE = {nse:.3f} | RMSE = {rmse:.2f} m³/s'

        fig.update_layout(
            title=title,
            height=1000,
            barmode='overlay',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black')
        )

        fig.update_yaxes(title_text='Precipitation (mm/hr)', row=1, col=1)
        fig.update_yaxes(title_text='Total Discharge (m³/s)', row=2, col=1)
        fig.update_yaxes(title_text='Discharge (m³/s)', row=3, col=1)
        fig.update_yaxes(title_text='Direct Runoff (m³/s)', row=4, col=1)
        fig.update_xaxes(title_text='Date', row=4, col=1)

        if save_path:
            fig.write_html(save_path)
            print(f"  Saved: {save_path}")

        return fig
    
    def plot_baseflow_separation(self, df, year=None, save_path=None):
        """
        Plot showing baseflow separation
        """
        if 'Q' not in df.columns:
            print("No observed discharge data for baseflow plot")
            return

        df_plot = df.copy()

        # Filter by year if specified
        if year:
            df_plot = df_plot[df_plot['Date Time'].dt.year == year]
        else:
            # Use most recent complete year
            years = df_plot['Date Time'].dt.year.unique()
            if len(years) > 1:
                year = years[-2]  # Second most recent
                df_plot = df_plot[df_plot['Date Time'].dt.year == year]

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Total discharge with components
        ax = axes[0]
        ax.plot(df_plot['Date Time'], df_plot['Q'], '-', color='black', linewidth=2.0, label='Total Observed Q', alpha=1.0)
        ax.plot(df_plot['Date Time'], df_plot['Q_baseflow_observed'], '-', color='dimgray', linewidth=2.0, label='Baseflow', alpha=1.0)
        ax.fill_between(df_plot['Date Time'], df_plot['Q_baseflow_observed'], alpha=0.15, color='gray', label='Baseflow component')
        ax.fill_between(df_plot['Date Time'], df_plot['Q_baseflow_observed'], df_plot['Q'], alpha=0.15, color='lightgray', label='Direct runoff component')
        ax.set_ylabel('Discharge (m³/s)', fontsize=11)
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.6, color='gray', linewidth=0.8)

        BFI = df_plot['BFI'].iloc[0] if 'BFI' in df_plot.columns else df_plot['Q_baseflow_observed'].mean() / df_plot['Q'].mean()
        ax.set_title(f'Baseflow Separation - {year}\nBFI = {BFI:.3f}', fontsize=12, fontweight='bold')

        # Direct runoff comparison
        ax = axes[1]
        Q_direct_observed = df_plot['Q'] - df_plot['Q_baseflow_observed']
        ax.plot(df_plot['Date Time'], Q_direct_observed, '-', color='black', linewidth=2.0, label='Observed Direct Runoff', alpha=1.0)
        ax.plot(df_plot['Date Time'], df_plot['Q_direct_simulated'], '--', color='dimgray', linewidth=2.0, label='Simulated Direct Runoff (UH)', alpha=1.0)
        ax.set_ylabel('Direct Runoff (m³/s)', fontsize=11)
        ax.set_xlabel('Date', fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.6, color='gray', linewidth=0.8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")

        return fig

    def plot_qdirect_waterlevel_yearly(self, df, year, water_level_col, save_path=None):
        """
        Plot yearly simulated direct runoff and water level with Spearman rank correlation.
        """
        df_year = df[df['Date Time'].dt.year == year].copy()

        if len(df_year) == 0:
            print(f"  No data for {year} (Q_direct vs water level)")
            return None, np.nan

        if 'Q_direct_simulated' in df_year.columns:
            q_direct = df_year['Q_direct_simulated']
        else:
            print(f"  No Q_direct_simulated column for {year}")
            return None, np.nan

        if water_level_col not in df_year.columns:
            print(f"  Water level column '{water_level_col}' not found for {year}")
            return None, np.nan

        wl = df_year[water_level_col]
        valid = q_direct.notna() & wl.notna()

        spearman_q_wl = np.nan
        if valid.sum() > 1:
            q_vals = q_direct[valid].values
            wl_vals = wl[valid].values
            if np.std(q_vals) > 0 and np.std(wl_vals) > 0:
                spearman_q_wl = spearmanr(q_vals, wl_vals)[0]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=df_year['Date Time'],
                y=q_direct,
                mode='lines',
                name='Q_direct simulated',
                line=dict(color='#000000', width=2.0)
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=df_year['Date Time'],
                y=wl,
                mode='lines',
                name=f'Water level ({water_level_col})',
                line=dict(color='#001f5b', width=2.0, dash='dash')
            ),
            secondary_y=True
        )

        title = f'Q_direct vs Water Level - {year}'
        if pd.notna(spearman_q_wl):
            title += f' | Spearman = {spearman_q_wl:.3f}'
        else:
            title += ' | Spearman = NA'

        fig.update_layout(
            title=title,
            height=500,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black')
        )

        fig.update_xaxes(
            title_text='Date',
            showline=True,
            linecolor='black',
            linewidth=2,
            mirror=True,
            ticks='outside',
            tickcolor='black',
            showgrid=True,
            gridcolor='rgba(120,120,120,0.30)',
            gridwidth=1,
        )
        fig.update_yaxes(
            title_text='Q_direct simulated (m³/s)',
            secondary_y=False,
            showline=True,
            linecolor='black',
            linewidth=2,
            mirror=True,
            ticks='outside',
            tickcolor='black',
            showgrid=True,
            gridcolor='rgba(120,120,120,0.30)',
            gridwidth=1,
        )
        fig.update_yaxes(
            title_text='Water level',
            secondary_y=True,
            showline=True,
            linecolor='black',
            linewidth=2,
            mirror=True,
            ticks='outside',
            tickcolor='black',
            showgrid=False,
        )

        if save_path:
            fig.write_html(save_path)
            print(f"  Saved: {save_path}")

            # Also export a high-resolution static image next to the HTML.
            img_path = os.path.splitext(save_path)[0] + ".png"
            try:
                fig.write_image(img_path, format='png', width=2200, height=1200, scale=2)
                print(f"  Saved high-res image: {img_path}")
            except Exception as e:
                print(f"  Could not save high-res image (install kaleido): {e}")

        return fig, spearman_q_wl

    def plot_multiyear_performance(self, df, year_range, save_path=None):
        """
        Plot all years in specified range (e.g., 2010-2021) in a single interactive plot
        
        Parameters:
        -----------
        df : DataFrame
            Data with simulations
        year_range : tuple or list
            (start_year, end_year) or list of years to plot
        save_path : str
            Path to save figure
        """
        # Handle year_range input
        if isinstance(year_range, tuple) and len(year_range) == 2:
            start_year, end_year = year_range
            years_to_plot = list(range(start_year, end_year + 1))
        else:
            years_to_plot = sorted(year_range) if hasattr(year_range, '__iter__') else [year_range]
        
        # Filter data to requested years
        df_multi = df[df['Date Time'].dt.year.isin(years_to_plot)].copy()
        
        if len(df_multi) == 0:
            print(f"  No data for years {years_to_plot}")
            return
        
        # Calculate overall statistics
        nse, rmse = None, None
        if 'Q' in df_multi.columns and 'Q_simulated_total' in df_multi.columns:
            total_metrics = self.calculate_performance_metrics(
                df_multi['Q'].values,
                df_multi['Q_simulated_total'].values
            )
            nse = total_metrics['NSE']
            rmse = total_metrics['RMSE']
        
        # Create interactive figure with 4 stacked subplots
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=(
                'Precipitation',
                'Total Discharge (Direct Runoff + Baseflow)',
                'Baseflow Separation',
                'Direct Runoff Only (UH Performance)'
            )
        )

        # 1. Precipitation (negative values for inverted visual style)
        fig.add_trace(
            go.Bar(
                x=df_multi['Date Time'],
                y=-df_multi['corr_precip'],
                name='Total Precip',
                marker_color='#0b3d91',
                opacity=0.95
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            go.Bar(
                x=df_multi['Date Time'],
                y=-df_multi['P_effective'],
                name='Effective Rainfall',
                marker_color='#001f5b',
                opacity=1.0
            ),
            row=1,
            col=1
        )

        # 2. Total discharge comparison
        if 'Q' in df_multi.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_multi['Date Time'],
                    y=df_multi['Q'],
                    mode='lines',
                    name='Observed Total Q',
                    line=dict(color='#000000', width=1.5),
                    opacity=0.8
                ),
                row=2,
                col=1
            )
        fig.add_trace(
            go.Scatter(
                x=df_multi['Date Time'],
                y=df_multi['Q_simulated_total'],
                mode='lines',
                name='Simulated Total Q',
                line=dict(color='#4a4a4a', width=1.5, dash='dash'),
                opacity=0.8
            ),
            row=2,
            col=1
        )

        # 3. Baseflow separation
        if 'Q' in df_multi.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_multi['Date Time'],
                    y=df_multi['Q'],
                    mode='lines',
                    name='Total Observed Q',
                    line=dict(color='#000000', width=1.5),
                    opacity=0.8
                ),
                row=3,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df_multi['Date Time'],
                    y=df_multi['Q_baseflow_observed'],
                    mode='lines',
                    name='Baseflow',
                    line=dict(color='#555555', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(140, 140, 140, 0.18)',
                    opacity=0.8
                ),
                row=3,
                col=1
            )

        # 4. Direct runoff comparison
        if 'Q' in df_multi.columns:
            Q_direct_obs = df_multi['Q'] - df_multi['Q_baseflow_observed']
            fig.add_trace(
                go.Scatter(
                    x=df_multi['Date Time'],
                    y=Q_direct_obs,
                    mode='lines',
                    name='Observed Direct Runoff',
                    line=dict(color='#000000', width=1.5),
                    opacity=0.8
                ),
                row=4,
                col=1
            )
        fig.add_trace(
            go.Scatter(
                x=df_multi['Date Time'],
                y=df_multi['Q_direct_simulated'],
                mode='lines',
                name='Simulated Direct Runoff (UH)',
                line=dict(color='#4a4a4a', width=1.5, dash='dash'),
                opacity=0.8
            ),
            row=4,
            col=1
        )

        year_str = f"{years_to_plot[0]}-{years_to_plot[-1]}"
        title = f'Unit Hydrograph Performance - Years {year_str} (Total Discharge)'
        if nse is not None:
            title += f' | NSE = {nse:.3f} | RMSE = {rmse:.2f} m³/s'

        fig.update_layout(
            title=title,
            height=1000,
            barmode='overlay',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black')
        )

        fig.update_yaxes(title_text='Precipitation (mm/hr)', row=1, col=1)
        fig.update_yaxes(title_text='Total Discharge (m³/s)', row=2, col=1)
        fig.update_yaxes(title_text='Discharge (m³/s)', row=3, col=1)
        fig.update_yaxes(title_text='Direct Runoff (m³/s)', row=4, col=1)
        fig.update_xaxes(title_text='Date', row=4, col=1)

        if save_path:
            fig.write_html(save_path)
            print(f"  Saved: {save_path}")

        return fig


def main():
    """
    Main execution
    """
    print("\n" + "="*70)
    print("SEASONAL UNIT HYDROGRAPH DISCHARGE SIMULATION")
    print("="*70)
    
    # ========================================================================
    # USER INPUTS
    # ========================================================================
    
    # Data file
    data_file = input("\nEnter data filename (e.g., data.xlsx): ").strip()
    if not data_file:
        print("Error: No filename provided")
        return
    
    # Watershed area
    area_input = input("Enter watershed area (km²): ").strip()
    WATERSHED_AREA = float(area_input)

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print("\n" + "─"*70)
    print("Loading data...")
    
    try:
        df = pd.read_excel(data_file)
    except:
        df = pd.read_csv(data_file)
    
    df['Date Time'] = pd.to_datetime(df['Date Time'])
    df = df.sort_values('Date Time').reset_index(drop=True)
    
    print(f"  Loaded {len(df)} records")
    print(f"  Date range: {df['Date Time'].min()} to {df['Date Time'].max()}")
    
    # ========================================================================
    # INITIALIZE SIMULATOR
    # ========================================================================
    
    simulator = SeasonalUHSimulator(WATERSHED_AREA)

    # Ask output directory, with a sensible default if left blank.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tp_rounded = int(round(simulator.Tp))
    default_output_dir = os.path.join(script_dir, f"continuous_simulation_{tp_rounded}")
    output_dir_input = input(
        f"Enter output directory (leave blank for default: {default_output_dir}): "
    ).strip()
    output_dir = output_dir_input if output_dir_input else default_output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    water_level_col = simulator.find_water_level_column(df)
    if water_level_col:
        print(f"  Found water level column: {water_level_col}")
    else:
        print("  No water level column found (expected names like water_level, WL, stage)")
    
    # ========================================================================
    # RUN SIMULATION
    # ========================================================================
    
    df_sim, stats = simulator.simulate_discharge(df, precip_col='corr_precip')
    
    # ========================================================================
    # SEASONAL ANALYSIS
    # ========================================================================
    
    print("\n" + "="*70)
    print("SEASONAL PERFORMANCE")
    print("="*70)
    
    seasons = {
        'Pre-monsoon': [3, 4, 5],
        'Monsoon': [6, 7, 8, 9],
        'Post-monsoon': [10, 11],
        'Winter': [12, 1, 2]
    }
    
    if 'Q_direct_observed' in df_sim.columns:
        for season_name, months in seasons.items():
            df_season = df_sim[df_sim['Date Time'].dt.month.isin(months)]
            
            if len(df_season) > 0:
                direct_metrics = simulator.calculate_performance_metrics(
                    df_season['Q_direct_observed'].values,
                    df_season['Q_direct_simulated'].values
                )
                
                # NSE on TOTAL DISCHARGE (for reference)
                if 'Q' in df_season.columns:
                    total_metrics = simulator.calculate_performance_metrics(
                        df_season['Q'].values,
                        df_season['Q_simulated_total'].values
                    )
                    nse_total = total_metrics['NSE']
                else:
                    nse_total = -999
                
                print(f"\n{season_name}:")
                print(f"  NSE (Direct Runoff): {direct_metrics['NSE']:.3f}  ← UH validation")
                print(f"  RMSE (Direct):       {direct_metrics['RMSE']:.2f} m³/s")
                if pd.notna(direct_metrics['Spearman']):
                    print(f"  Spearman (Direct):   {direct_metrics['Spearman']:.3f}")
                if nse_total > -999:
                    print(f"  NSE (Total Q):       {nse_total:.3f}  ← Reference only")
    
    # ========================================================================
    # YEARLY PLOTS
    # ========================================================================
    
    print("\n" + "="*70)
    print("CREATING INTERACTIVE YEARLY PLOTS")
    print("="*70)
    
    years = sorted(df_sim['Date Time'].dt.year.unique())
    
    yearly_stats = []
    yearly_qdirect_wl_stats = []
    
    for year in years:
        print(f"\nYear {year}...")
        
        # Calculate yearly statistics
        df_year = df_sim[df_sim['Date Time'].dt.year == year]
        
        if 'Q' in df_year.columns and 'Q_simulated_total' in df_year.columns and len(df_year) > 0:
            nse = -999
            rmse = -999
            records = len(df_year)
            
            # Total discharge statistics only (Q observed vs Q simulated total)
            total_metrics = simulator.calculate_performance_metrics(
                df_year['Q'].values,
                df_year['Q_simulated_total'].values
            )
            nse = total_metrics['NSE']
            rmse = total_metrics['RMSE']
            
            yearly_stats.append({
                'Year': year,
                'NSE': nse,
                'RMSE': rmse,
                'Records': records
            })
            
            message = f"  NSE: {nse:.3f} | RMSE: {rmse:.2f} m³/s"
            print(message)
        
        # Create yearly plot
        simulator.plot_yearly_performance(
            df_sim,
            year=year,
            save_path=os.path.join(output_dir, f'discharge_simulation_year_{year}.html')
        )

        # Create yearly Q_direct vs water level plot (if water level exists)
        if water_level_col and water_level_col in df_sim.columns:
            _, spearman_q_wl = simulator.plot_qdirect_waterlevel_yearly(
                df_sim,
                year=year,
                water_level_col=water_level_col,
                save_path=os.path.join(output_dir, f'qdirect_waterlevel_year_{year}.html')
            )
            yearly_qdirect_wl_stats.append({
                'Year': year,
                'Spearman_Qdirect_WaterLevel': spearman_q_wl
            })
    
    # Create yearly statistics summary
    if yearly_stats:
        df_yearly = pd.DataFrame(yearly_stats)
        
        print("\n" + "="*70)
        print("YEARLY PERFORMANCE SUMMARY")
        print("="*70)
        print(df_yearly.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
        
        # Save to CSV
        yearly_stats_path = os.path.join(output_dir, 'yearly_statistics.csv')
        df_yearly.to_csv(yearly_stats_path, index=False)
        print(f"\n✓ Saved: {yearly_stats_path}")

    if yearly_qdirect_wl_stats:
        df_qwl = pd.DataFrame(yearly_qdirect_wl_stats)
        qwl_stats_path = os.path.join(output_dir, 'yearly_qdirect_waterlevel_spearman.csv')
        df_qwl.to_csv(qwl_stats_path, index=False)
        print(f"✓ Saved: {qwl_stats_path}")
    
    # ========================================================================
    # MULTI-YEAR PLOT (All years 2010-2021)
    # ========================================================================
    
    print("\n" + "="*70)
    print("CREATING MULTI-YEAR PLOT (2010-2021)")
    print("="*70)
    
    # Filter available years in the 2010-2021 range
    available_years = sorted(df_sim['Date Time'].dt.year.unique())
    years_in_range = [y for y in range(2010, 2022) if y in available_years]
    
    if len(years_in_range) > 0:
        print(f"Available years in range 2010-2021: {years_in_range}")
        simulator.plot_multiyear_performance(
            df_sim,
            year_range=(2010, 2021),
            save_path=os.path.join(output_dir, 'discharge_simulation_2010_2021.html')
        )
    else:
        print("  No data available for 2010-2021 range")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    print("\n" + "─"*70)
    print("Saving results...")
    
    # Save simulated timeseries
    output_cols = ['Date Time', 'corr_precip', 'P_effective', 'phi_index']
    if 'Q' in df_sim.columns:
        output_cols.append('Q')
    output_cols.extend(['Q_direct_observed', 'Q_direct_simulated'])
    if 'Q_baseflow_observed' in df_sim.columns:
        output_cols.extend(['Q_baseflow_observed', 'Q_simulated_total'])
    if 'residual' in df_sim.columns:
        output_cols.append('residual')
    
    df_output = df_sim[output_cols].copy()
    timeseries_path = os.path.join(output_dir, 'simulated_discharge_timeseries.csv')
    df_output.to_csv(timeseries_path, index=False)
    print(f"  Saved: {timeseries_path}")
    
    # Save statistics summary
    if stats:
        stats_path = os.path.join(output_dir, 'simulation_statistics.txt')
        with open(stats_path, 'w') as f:
            f.write("DISCHARGE SIMULATION STATISTICS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Watershed Area: {WATERSHED_AREA:.2f} km²\n")
            f.write(f"Time to Peak (Tp): {simulator.Tp:.1f} hours\n")
            f.write(f"Time Base (Tb): {simulator.Tb:.1f} hours\n")
            if 'BFI' in df_sim.columns:
                f.write(f"Baseflow Index (BFI): {df_sim['BFI'].iloc[0]:.3f}\n")
            f.write("\n")
            f.write("="*70 + "\n")
            f.write("UNIT HYDROGRAPH VALIDATION (Direct Runoff Comparison):\n")
            f.write("="*70 + "\n")
            f.write("This compares: Q_direct_simulated vs Q_direct (from input file)\n\n")
            f.write(f"  NSE (Direct Runoff):   {stats['NSE_direct']:.3f}\n")
            f.write(f"  RMSE (Direct Runoff):  {stats['RMSE_direct']:.2f} m³/s\n")
            f.write(f"  PBIAS (Direct Runoff): {stats['PBIAS_direct']:.1f}%\n\n")
            if pd.notna(stats['Spearman_direct']):
                f.write(f"  Spearman (Direct):     {stats['Spearman_direct']:.3f}\n\n")
            if stats['NSE_total'] > -999:
                f.write("="*70 + "\n")
                f.write("TOTAL DISCHARGE Performance (Reference Only):\n")
                f.write("="*70 + "\n")
                f.write("This includes baseflow uncertainty, less reliable for UH validation.\n\n")
                f.write(f"  NSE (Total Q):         {stats['NSE_total']:.3f}\n")
        print(f"  Saved: {stats_path}")
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("\nInteractive Yearly Performance:")
    for year in years:
        print(f"  - discharge_simulation_year_{year}.html")
        if water_level_col:
            print(f"  - qdirect_waterlevel_year_{year}.html")
            print(f"  - qdirect_waterlevel_year_{year}.png")
    print("\nMulti-Year Plot:")
    print("  - discharge_simulation_2010_2021.html")
    print("\nData Files:")
    print("  - simulated_discharge_timeseries.csv")
    print("  - yearly_statistics.csv")
    if yearly_qdirect_wl_stats:
        print("  - yearly_qdirect_waterlevel_spearman.csv")
    if stats:
        print("  - simulation_statistics.txt")
    print("\nNote: Baseflow calculated from observed Q minus observed Q_direct")
    print("="*70)


if __name__ == "__main__":
    main()
