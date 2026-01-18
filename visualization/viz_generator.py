import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom
from config import Config

# Make SHAP optional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class VisualizationGenerator:
    """
    Generates all possible relevant visualizations for a dataset
    """
    
    def __init__(self, df, output_dir="graphs"):
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.generated_graphs = []
        
        # Analyze data types - numpy 2.x compatible
        self.numeric_cols = df.select_dtypes(include='number').columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include='datetime64').columns.tolist()
        
    def generate_all(self):
        """Generate all relevant visualizations"""
        print("Generating visualizations...")
        
        # Basic statistical plots
        self._generate_distributions()
        self._generate_correlations()
        self._generate_box_plots()
        
        # Advanced plots
        if len(self.numeric_cols) >= 2:
            self._generate_pca()
            self._generate_tsne()
            self._generate_pair_plots()
        
        # ML explainability (if suitable)
        if len(self.numeric_cols) >= 3:
            self._generate_shap_placeholder()
            self._generate_som()
        
        # Time series if applicable
        if self.datetime_cols:
            self._generate_time_series()
        
        # Categorical analysis
        if self.categorical_cols:
            self._generate_categorical_plots()
        
        print(f"Generated {len(self.generated_graphs)} graphs")
        return self.generated_graphs
    
    def _save_figure(self, fig, name):
        """Save figure and track it"""
        filepath = self.output_dir / f"{name}.png"
        fig.savefig(filepath, dpi=Config.FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        self.generated_graphs.append(str(filepath))
        return filepath
    
    def _generate_distributions(self):
        """Histograms for numeric columns"""
        for col in self.numeric_cols[:10]:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                self.df[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                self._save_figure(fig, f"dist_{col}")
            except Exception as e:
                print(f"Distribution plot for {col} failed: {e}")
    
    def _generate_correlations(self):
        """Correlation heatmap"""
        if len(self.numeric_cols) < 2:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            corr = self.df[self.numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                        center=0, ax=ax, square=True, linewidths=0.5)
            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
            self._save_figure(fig, "correlation_heatmap")
        except Exception as e:
            print(f"Correlation heatmap failed: {e}")
    
    def _generate_box_plots(self):
        """Box plots for numeric columns"""
        for col in self.numeric_cols[:8]:
            try:
                fig, ax = plt.subplots(figsize=(6, 6))
                self.df.boxplot(column=col, ax=ax)
                ax.set_title(f'Box Plot: {col}', fontsize=12, fontweight='bold')
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3, axis='y')
                self._save_figure(fig, f"box_{col}")
            except Exception as e:
                print(f"Box plot for {col} failed: {e}")
    
    def _generate_pca(self):
        """PCA visualization"""
        try:
            X = self.df[self.numeric_cols].dropna()
            if len(X) < 10:
                return
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(components[:, 0], components[:, 1], 
                               alpha=0.6, c=range(len(components)), 
                               cmap='viridis', s=50)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
            ax.set_title('PCA Analysis', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax)
            self._save_figure(fig, "pca_analysis")
        except Exception as e:
            print(f"PCA generation failed: {e}")
    
    def _generate_tsne(self):
        """t-SNE visualization"""
        try:
            X = self.df[self.numeric_cols].dropna()
            if len(X) < 30 or len(X) > 10000:
                return
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Sample if too large
            if len(X_scaled) > 1000:
                from sklearn.utils import resample
                X_scaled = resample(X_scaled, n_samples=1000, random_state=42)
            
            perplexity = min(30, len(X_scaled) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
            embedded = tsne.fit_transform(X_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1], 
                               alpha=0.6, c=range(len(embedded)), 
                               cmap='plasma', s=50)
            ax.set_title('t-SNE Visualization', fontsize=14, fontweight='bold')
            ax.set_xlabel('Dimension 1', fontsize=12)
            ax.set_ylabel('Dimension 2', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax)
            self._save_figure(fig, "tsne_analysis")
        except Exception as e:
            print(f"t-SNE generation failed: {e}")
    
    def _generate_som(self):
        """Self-Organizing Map"""
        try:
            X = self.df[self.numeric_cols].dropna()
            if len(X) < 50:
                return
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            som_shape = (10, 10)
            som = MiniSom(som_shape[0], som_shape[1], X_scaled.shape[1], 
                         sigma=1.0, learning_rate=0.5, random_seed=42)
            som.random_weights_init(X_scaled)
            som.train_random(X_scaled, 1000, verbose=False)
            
            fig, ax = plt.subplots(figsize=(10, 10))
            distance_map = som.distance_map().T
            im = ax.imshow(distance_map, cmap='bone_r', origin='lower')
            ax.set_title('Self-Organizing Map (SOM) - Distance Map', 
                        fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Distance')
            self._save_figure(fig, "som_distance_map")
        except Exception as e:
            print(f"SOM generation failed: {e}")
    
    def _generate_shap_placeholder(self):
        """SHAP visualization (optional)"""
        if not SHAP_AVAILABLE:
            print("SHAP not available - skipping")
            return
        pass
    
    def _generate_pair_plots(self):
        """Pair plots for top numeric columns"""
        if len(self.numeric_cols) > 6:
            cols_to_plot = self.numeric_cols[:6]
        else:
            cols_to_plot = self.numeric_cols
        
        if len(cols_to_plot) < 2:
            return
        
        try:
            df_plot = self.df[cols_to_plot].dropna()
            if len(df_plot) > 1000:
                df_plot = df_plot.sample(n=1000, random_state=42)
            
            fig = sns.pairplot(df_plot, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
            fig.fig.suptitle('Pair Plot Analysis', y=1.02, fontsize=14, fontweight='bold')
            self._save_figure(fig.fig, "pair_plot")
        except Exception as e:
            print(f"Pair plot failed: {e}")
    
    def _generate_time_series(self):
        """Time series plots if datetime columns exist"""
        for date_col in self.datetime_cols:
            for num_col in self.numeric_cols[:5]:
                try:
                    df_sorted = self.df.sort_values(date_col)
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(df_sorted[date_col], df_sorted[num_col], 
                           marker='o', markersize=3, alpha=0.7, linewidth=1.5)
                    ax.set_xlabel(date_col, fontsize=12)
                    ax.set_ylabel(num_col, fontsize=12)
                    ax.set_title(f'{num_col} over {date_col}', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45, ha='right')
                    self._save_figure(fig, f"timeseries_{date_col}_{num_col}")
                except Exception as e:
                    print(f"Time series plot failed: {e}")
    
    def _generate_categorical_plots(self):
        """Categorical analysis"""
        for cat_col in self.categorical_cols[:5]:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts = self.df[cat_col].value_counts().head(15)
                value_counts.plot(kind='bar', ax=ax, alpha=0.7, edgecolor='black')
                ax.set_title(f'Distribution of {cat_col}', fontsize=12, fontweight='bold')
                ax.set_xlabel(cat_col, fontsize=11)
                ax.set_ylabel('Count', fontsize=11)
                ax.grid(True, alpha=0.3, axis='y')
                plt.xticks(rotation=45, ha='right')
                self._save_figure(fig, f"cat_{cat_col}")
            except Exception as e:
                print(f"Categorical plot failed: {e}")
        
        # Categorical vs Numeric
        if self.categorical_cols and self.numeric_cols:
            for cat_col in self.categorical_cols[:3]:
                for num_col in self.numeric_cols[:3]:
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        grouped = self.df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
                        grouped.head(10).plot(kind='bar', ax=ax, alpha=0.7, edgecolor='black')
                        ax.set_title(f'Average {num_col} by {cat_col}', 
                                   fontsize=12, fontweight='bold')
                        ax.set_xlabel(cat_col, fontsize=11)
                        ax.set_ylabel(f'Mean {num_col}', fontsize=11)
                        ax.grid(True, alpha=0.3, axis='y')
                        plt.xticks(rotation=45, ha='right')
                        self._save_figure(fig, f"catnum_{cat_col}_{num_col}")
                    except Exception as e:
                        print(f"Cat-Num plot failed: {e}")
