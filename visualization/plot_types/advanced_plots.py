import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
from typing import Optional

# Optional UMAP import
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def plot_pca(df: pd.DataFrame,
            numeric_columns: list,
            output_path: Path,
            n_components: int = 2,
            color_by: Optional[str] = None) -> Path:
    """
    Create PCA visualization
    
    Args:
        df: Input dataframe
        numeric_columns: Numeric columns for PCA
        output_path: Path to save figure
        n_components: Number of components (2 or 3)
        color_by: Optional column for coloring points
        
    Returns:
        Path to saved figure
    """
    # Prepare data
    X = df[numeric_columns].dropna()
    
    if len(X) < 10:
        raise ValueError("Insufficient data points for PCA")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=min(n_components, len(numeric_columns)))
    components = pca.fit_transform(X_scaled)
    
    # Plot
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if color_by and color_by in df.columns:
            colors = df.loc[X.index, color_by]
            scatter = ax.scatter(components[:, 0], components[:, 1],
                               c=colors, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            scatter = ax.scatter(components[:, 0], components[:, 1],
                               alpha=0.6, s=50, c=range(len(components)),
                               cmap='viridis')
            plt.colorbar(scatter, ax=ax)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                     fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
                     fontsize=12)
        ax.set_title('PCA - Principal Component Analysis',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    else:  # 3D
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(components[:, 0], components[:, 1], components[:, 2],
                           c=range(len(components)), cmap='viridis',
                           alpha=0.6, s=50)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=10)
        ax.set_title('PCA - 3D Projection', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, shrink=0.6)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_tsne(df: pd.DataFrame,
             numeric_columns: list,
             output_path: Path,
             perplexity: int = 30,
             n_iter: int = 1000) -> Path:
    """
    Create t-SNE visualization
    
    Args:
        df: Input dataframe
        numeric_columns: Numeric columns
        output_path: Path to save figure
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        
    Returns:
        Path to saved figure
    """
    # Prepare data
    X = df[numeric_columns].dropna()
    
    if len(X) < 30:
        raise ValueError("t-SNE requires at least 30 samples")
    
    # Limit samples for performance
    if len(X) > 5000:
        X = X.sample(n=5000, random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Adjust perplexity
    perplexity = min(perplexity, len(X) - 1)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, 
                n_iter=n_iter, random_state=42, verbose=0)
    embedded = tsne.fit_transform(X_scaled)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                        c=range(len(embedded)), cmap='plasma',
                        alpha=0.6, s=50)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(f't-SNE Visualization (perplexity={perplexity})',
                 fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_umap(df: pd.DataFrame,
             numeric_columns: list,
             output_path: Path,
             n_neighbors: int = 15,
             min_dist: float = 0.1) -> Path:
    """
    Create UMAP visualization (if available)
    
    Args:
        df: Input dataframe
        numeric_columns: Numeric columns
        output_path: Path to save figure
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        
    Returns:
        Path to saved figure
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not installed. Install with: pip install umap-learn")
    
    # Prepare data
    X = df[numeric_columns].dropna()
    
    if len(X) < 10:
        raise ValueError("Insufficient data for UMAP")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # UMAP
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=range(len(embedding)), cmap='Spectral',
                        alpha=0.6, s=50)
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title('UMAP Projection', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_som(df: pd.DataFrame,
            numeric_columns: list,
            output_path: Path,
            grid_size: tuple = (10, 10),
            n_iterations: int = 1000) -> Path:
    """
    Create Self-Organizing Map visualization
    
    Args:
        df: Input dataframe
        numeric_columns: Numeric columns
        output_path: Path to save figure
        grid_size: SOM grid dimensions
        n_iterations: Training iterations
        
    Returns:
        Path to saved figure
    """
    # Prepare data
    X = df[numeric_columns].dropna()
    
    if len(X) < 50:
        raise ValueError("SOM requires at least 50 samples")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize SOM
    som = MiniSom(grid_size[0], grid_size[1], X_scaled.shape[1],
                 sigma=1.0, learning_rate=0.5, random_seed=42)
    
    # Train
    som.random_weights_init(X_scaled)
    som.train_random(X_scaled, n_iterations, verbose=False)
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Distance map
    distance_map = som.distance_map().T
    im1 = ax1.imshow(distance_map, cmap='bone_r', origin='lower')
    ax1.set_title('SOM Distance Map (U-Matrix)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Average Distance')
    
    # Hit map (frequency)
    hit_map = np.zeros(grid_size)
    for x in X_scaled:
        winner = som.winner(x)
        hit_map[winner] += 1
    
    im2 = ax2.imshow(hit_map.T, cmap='YlOrRd', origin='lower')
    ax2.set_title('SOM Hit Map (Frequency)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Number of Hits')
    
    fig.suptitle('Self-Organizing Map Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path
