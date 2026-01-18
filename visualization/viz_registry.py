from typing import Dict, List, Callable
import pandas as pd
from dataclasses import dataclass

@dataclass
class PlotConfig:
    """Configuration for a plot type"""
    name: str
    func: Callable
    requires_numeric: int = 0
    requires_categorical: int = 0
    requires_datetime: int = 0
    description: str = ""
    priority: int = 5

class VisualizationRegistry:
    """
    Central registry for all available plot types
    """
    
    def __init__(self):
        self.plots: Dict[str, PlotConfig] = {}
    
    def register(self, plot_config: PlotConfig):
        """Register a new plot type"""
        self.plots[plot_config.name] = plot_config
    
    def get_applicable_plots(self, df: pd.DataFrame) -> List[PlotConfig]:
        """Get list of plots applicable to the given dataframe"""
        
        n_numeric = len(df.select_dtypes(include='number').columns)
        n_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
        n_datetime = len(df.select_dtypes(include='datetime64').columns)
        
        applicable = []
        
        for plot_name, plot_config in self.plots.items():
            if (n_numeric >= plot_config.requires_numeric and
                n_categorical >= plot_config.requires_categorical and
                n_datetime >= plot_config.requires_datetime):
                applicable.append(plot_config)
        
        applicable.sort(key=lambda x: x.priority, reverse=True)
        
        return applicable
    
    def get_plot(self, name: str) -> PlotConfig:
        """Get specific plot configuration by name"""
        return self.plots.get(name)
    
    def list_all(self) -> List[str]:
        """List all registered plot names"""
        return list(self.plots.keys())


# Global registry instance
registry = VisualizationRegistry()
