import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List

def plot_correlation_heatmap(df: pd.DataFrame,
                             numeric_columns: List[str],
                             output_path: Path,
                             method: str = 'pearson',
                             annot: bool = True) -> Path:
    """
    Create correlation heatmap
    
    Args:
        df: Input dataframe
        numeric_columns: Numeric columns to include
        output_path: Path to save figure
        method: Correlation method ('pearson', 'spearman', 'kendall')
        annot: Whether to annotate cells with values
        
    Returns:
        Path to saved figure
    """
    # Calculate correlation
    corr = df[numeric_columns].corr(method=method)
    
    # Determine figure size based on number of columns
    n_cols = len(numeric_columns)
    fig_size = max(8, min(n_cols * 0.8, 20))
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # Create mask for upper triangle (optional)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    # Plot heatmap
    sns.heatmap(corr,
                mask=mask,
                annot=annot if n_cols <= 15 else False,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title(f'Correlation Heatmap ({method.capitalize()})',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_scatter_matrix(df: pd.DataFrame,
                       columns: List[str],
                       output_path: Path,
                       max_cols: int = 6,
                       hue: Optional[str] = None) -> Path:
    """
    Create scatter matrix (pair plot alternative)
    
    Args:
        df: Input dataframe
        columns: Columns to include
        output_path: Path to save figure
        max_cols: Maximum columns to plot
        hue: Optional column for color coding
        
    Returns:
        Path to saved figure
    """
    columns = columns[:max_cols]
    
    if hue and hue not in columns:
        plot_data = df[columns + [hue]]
    else:
        plot_data = df[columns]
        hue = None
    
    # Create scatter matrix
    from pandas.plotting import scatter_matrix
    
    n_cols = len(columns)
    fig_size = max(10, n_cols * 2.5)
    
    fig, axes = plt.subplots(n_cols, n_cols, figsize=(fig_size, fig_size))
    
    scatter_matrix(plot_data[columns], 
                   alpha=0.6,
                   figsize=(fig_size, fig_size),
                   diagonal='kde',
                   ax=axes)
    
    plt.suptitle('Scatter Matrix', fontsize=16, fontweight='bold', y=0.995)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_pairplot(df: pd.DataFrame,
                 columns: List[str],
                 output_path: Path,
                 max_cols: int = 6,
                 hue: Optional[str] = None) -> Path:
    """
    Create seaborn pairplot
    
    Args:
        df: Input dataframe
        columns: Columns to include
        output_path: Path to save figure
        max_cols: Maximum columns to plot
        hue: Optional column for color coding
        
    Returns:
        Path to saved figure
    """
    columns = columns[:max_cols]
    
    # Prepare data
    if hue and hue in df.columns:
        plot_data = df[columns + [hue]].dropna()
    else:
        plot_data = df[columns].dropna()
        hue = None
    
    # Limit data points if too many
    if len(plot_data) > 1000:
        plot_data = plot_data.sample(n=1000, random_state=42)
    
    # Create pairplot
    g = sns.pairplot(plot_data,
                     hue=hue,
                     diag_kind='kde',
                     plot_kws={'alpha': 0.6, 's': 30},
                     diag_kws={'alpha': 0.7})
    
    g.fig.suptitle('Pair Plot Analysis', fontsize=16, fontweight='bold', y=1.01)
    
    g.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(g.fig)
    
    return output_path
