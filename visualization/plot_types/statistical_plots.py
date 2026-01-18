import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

def plot_histogram(df: pd.DataFrame, 
                  column: str,
                  output_path: Path,
                  bins: int = 30,
                  kde: bool = True) -> Path:
    """
    Create histogram with optional KDE overlay
    
    Args:
        df: Input dataframe
        column: Column to plot
        output_path: Path to save figure
        bins: Number of bins
        kde: Whether to overlay KDE
        
    Returns:
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(df[column].dropna(), bins=bins, alpha=0.7, 
            edgecolor='black', density=kde, label='Distribution')
    
    # KDE overlay
    if kde:
        df[column].dropna().plot.kde(ax=ax, linewidth=2, 
                                     color='red', label='KDE')
    
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Frequency' if not kde else 'Density', fontsize=12)
    ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_boxplot(df: pd.DataFrame,
                columns: List[str],
                output_path: Path,
                orientation: str = 'v') -> Path:
    """
    Create box plot for numeric columns
    
    Args:
        df: Input dataframe
        columns: Columns to include
        output_path: Path to save figure
        orientation: 'v' for vertical, 'h' for horizontal
        
    Returns:
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(max(10, len(columns) * 1.5), 6))
    
    data_to_plot = [df[col].dropna() for col in columns]
    
    bp = ax.boxplot(data_to_plot, 
                    labels=columns,
                    patch_artist=True,
                    showmeans=True,
                    meanline=True,
                    vert=(orientation == 'v'))
    
    # Customize colors
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    if orientation == 'v':
        ax.set_ylabel('Value', fontsize=12)
        plt.xticks(rotation=45, ha='right')
    else:
        ax.set_xlabel('Value', fontsize=12)
    
    ax.set_title('Box Plot Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y' if orientation == 'v' else 'x')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_violin(df: pd.DataFrame,
               columns: List[str],
               output_path: Path,
               max_cols: int = 6) -> Path:
    """
    Create violin plots for numeric columns
    
    Args:
        df: Input dataframe
        columns: Columns to plot
        output_path: Path to save figure
        max_cols: Maximum columns to plot
        
    Returns:
        Path to saved figure
    """
    columns = columns[:max_cols]
    
    fig, ax = plt.subplots(figsize=(max(10, len(columns) * 1.5), 6))
    
    # Prepare data in long format
    df_long = df[columns].melt(var_name='Variable', value_name='Value')
    
    sns.violinplot(data=df_long, x='Variable', y='Value', 
                   ax=ax, inner='box', palette='Set2')
    
    ax.set_xlabel('')
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Violin Plot - Distribution Comparison', 
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_density(df: pd.DataFrame,
                columns: List[str],
                output_path: Path,
                max_cols: int = 6) -> Path:
    """
    Create density plots for multiple columns
    
    Args:
        df: Input dataframe
        columns: Columns to plot
        output_path: Path to save figure
        max_cols: Maximum columns to plot
        
    Returns:
        Path to saved figure
    """
    columns = columns[:max_cols]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col in columns:
        df[col].dropna().plot.kde(ax=ax, linewidth=2, label=col, alpha=0.7)
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Density Plot Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path
