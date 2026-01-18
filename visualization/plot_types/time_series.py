import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

def plot_time_series(df: pd.DataFrame,
                    datetime_column: str,
                    value_columns: List[str],
                    output_path: Path,
                    rolling_window: Optional[int] = None) -> Path:
    """
    Create time series plot with optional rolling average
    
    Args:
        df: Input dataframe
        datetime_column: Column containing datetime values
        value_columns: List of columns to plot over time
        output_path: Path to save figure
        rolling_window: Optional rolling average window size
        
    Returns:
        Path to saved figure
    """
    # Sort by datetime
    df_sorted = df.sort_values(datetime_column).copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot each value column
    for col in value_columns[:5]:  # Limit to 5 series for clarity
        ax.plot(df_sorted[datetime_column], df_sorted[col],
               marker='o', markersize=3, alpha=0.7, label=col, linewidth=1.5)
        
        # Add rolling average if specified
        if rolling_window and rolling_window > 1:
            rolling = df_sorted[col].rolling(window=rolling_window, min_periods=1).mean()
            ax.plot(df_sorted[datetime_column], rolling,
                   linestyle='--', alpha=0.8, linewidth=2,
                   label=f'{col} (MA-{rolling_window})')
    
    # Formatting
    ax.set_xlabel(datetime_column, fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Time Series Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_seasonal_decompose(df: pd.DataFrame,
                           datetime_column: str,
                           value_column: str,
                           output_path: Path,
                           period: int = 12,
                           model: str = 'additive') -> Path:
    """
    Plot seasonal decomposition of time series
    
    Args:
        df: Input dataframe
        datetime_column: Column containing datetime values
        value_column: Column to decompose
        output_path: Path to save figure
        period: Seasonal period (e.g., 12 for monthly data with yearly seasonality)
        model: 'additive' or 'multiplicative'
        
    Returns:
        Path to saved figure
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")
    
    # Prepare time series
    df_sorted = df.sort_values(datetime_column).copy()
    df_sorted.set_index(datetime_column, inplace=True)
    ts = df_sorted[value_column].dropna()
    
    # Check if we have enough data
    min_required = 2 * period
    if len(ts) < min_required:
        raise ValueError(f"Need at least {min_required} observations for period={period}. Got {len(ts)}")
    
    # Perform decomposition
    try:
        decomposition = seasonal_decompose(ts, model=model, period=period, extrapolate_trend='freq')
    except Exception as e:
        raise ValueError(f"Decomposition failed: {str(e)}")
    
    # Create plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    # Observed
    decomposition.observed.plot(ax=axes[0], color='blue', linewidth=1.5)
    axes[0].set_ylabel('Observed', fontsize=11)
    axes[0].set_title('Original Time Series', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    decomposition.trend.plot(ax=axes[1], color='red', linewidth=2)
    axes[1].set_ylabel('Trend', fontsize=11)
    axes[1].set_title('Trend Component', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    decomposition.seasonal.plot(ax=axes[2], color='green', linewidth=1.5)
    axes[2].set_ylabel('Seasonal', fontsize=11)
    axes[2].set_title('Seasonal Component', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    decomposition.resid.plot(ax=axes[3], color='purple', linewidth=1)
    axes[3].set_ylabel('Residual', fontsize=11)
    axes[3].set_title('Residual Component', fontsize=11, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel('Time', fontsize=11)
    
    # Overall title
    fig.suptitle(f'Seasonal Decomposition - {value_column} ({model.capitalize()} Model)',
                 fontsize=14, fontweight='bold', y=0.995)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_autocorrelation(df: pd.DataFrame,
                        datetime_column: str,
                        value_column: str,
                        output_path: Path,
                        lags: int = 40) -> Path:
    """
    Plot autocorrelation (ACF) and partial autocorrelation (PACF)
    
    Args:
        df: Input dataframe
        datetime_column: Column containing datetime values
        value_column: Column to analyze
        output_path: Path to save figure
        lags: Number of lags to display
        
    Returns:
        Path to saved figure
    """
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")
    
    # Prepare time series
    df_sorted = df.sort_values(datetime_column).copy()
    df_sorted.set_index(datetime_column, inplace=True)
    ts = df_sorted[value_column].dropna()
    
    # Adjust lags if necessary
    max_lags = len(ts) - 1
    if lags >= max_lags:
        lags = max(10, max_lags // 2)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # ACF plot
    try:
        plot_acf(ts, lags=lags, ax=ax1, alpha=0.05)
        ax1.set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Lag', fontsize=11)
        ax1.set_ylabel('ACF', fontsize=11)
        ax1.grid(True, alpha=0.3)
    except Exception as e:
        ax1.text(0.5, 0.5, f'ACF plot failed: {str(e)}', 
                ha='center', va='center', transform=ax1.transAxes)
    
    # PACF plot
    try:
        plot_pacf(ts, lags=lags, ax=ax2, alpha=0.05, method='ywm')
        ax2.set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Lag', fontsize=11)
        ax2.set_ylabel('PACF', fontsize=11)
        ax2.grid(True, alpha=0.3)
    except Exception as e:
        ax2.text(0.5, 0.5, f'PACF plot failed: {str(e)}', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # Overall title
    fig.suptitle(f'Autocorrelation Analysis - {value_column}',
                 fontsize=14, fontweight='bold', y=0.995)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_trend_analysis(df: pd.DataFrame,
                       datetime_column: str,
                       value_column: str,
                       output_path: Path) -> Path:
    """
    Plot time series with trend line and confidence interval
    
    Args:
        df: Input dataframe
        datetime_column: Column containing datetime values
        value_column: Column to analyze
        output_path: Path to save figure
        
    Returns:
        Path to saved figure
    """
    from scipy import stats
    
    # Sort by datetime
    df_sorted = df.sort_values(datetime_column).copy()
    df_sorted = df_sorted[df_sorted[value_column].notna()]
    
    if len(df_sorted) < 3:
        raise ValueError("Need at least 3 data points for trend analysis")
    
    # Convert datetime to numeric for regression
    df_sorted['_time_numeric'] = (df_sorted[datetime_column] - df_sorted[datetime_column].min()).dt.total_seconds()
    
    # Linear regression
    x = df_sorted['_time_numeric'].values
    y = df_sorted[value_column].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    trend_line = slope * x + intercept
    
    # Calculate confidence interval
    predict_error = np.sqrt(np.sum((y - trend_line) ** 2) / (len(y) - 2))
    confidence = 1.96 * predict_error  # 95% CI
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Original data
    ax.plot(df_sorted[datetime_column], y, marker='o', markersize=4,
           alpha=0.6, linewidth=1, label='Actual', color='blue')
    
    # Trend line
    ax.plot(df_sorted[datetime_column], trend_line, 
           linestyle='--', linewidth=2, color='red',
           label=f'Trend (R²={r_value**2:.3f})')
    
    # Confidence interval
    ax.fill_between(df_sorted[datetime_column],
                    trend_line - confidence,
                    trend_line + confidence,
                    alpha=0.2, color='red',
                    label='95% Confidence Interval')
    
    # Formatting
    ax.set_xlabel(datetime_column, fontsize=12)
    ax.set_ylabel(value_column, fontsize=12)
    ax.set_title(f'Trend Analysis - {value_column}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add statistics text
    stats_text = f'Slope: {slope:.4e}\nP-value: {p_value:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_multiple_time_series(df: pd.DataFrame,
                              datetime_column: str,
                              value_columns: List[str],
                              output_path: Path,
                              normalize: bool = False) -> Path:
    """
    Plot multiple time series on separate subplots for comparison
    
    Args:
        df: Input dataframe
        datetime_column: Column containing datetime values
        value_columns: List of columns to plot
        output_path: Path to save figure
        normalize: Whether to normalize each series to 0-1 range
        
    Returns:
        Path to saved figure
    """
    # Sort by datetime
    df_sorted = df.sort_values(datetime_column).copy()
    
    # Limit to 6 series
    value_columns = value_columns[:6]
    n_plots = len(value_columns)
    
    # Create subplots
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, col in enumerate(value_columns):
        data = df_sorted[col].copy()
        
        if normalize:
            # Normalize to 0-1 range
            min_val = data.min()
            max_val = data.max()
            if max_val > min_val:
                data = (data - min_val) / (max_val - min_val)
        
        axes[idx].plot(df_sorted[datetime_column], data,
                      marker='o', markersize=3, alpha=0.7, linewidth=1.5)
        axes[idx].set_ylabel(col if not normalize else f'{col} (normalized)',
                           fontsize=11)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_title(f'{col} over Time', fontsize=11, fontweight='bold')
    
    axes[-1].set_xlabel(datetime_column, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    title = 'Multiple Time Series Comparison'
    if normalize:
        title += ' (Normalized)'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path
