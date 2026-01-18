import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class DataValidator:
    """
    Validate and clean uploaded data
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues = []
        self.warnings = []
    
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Run all validation checks
        
        Returns:
            (is_valid, errors, warnings)
        """
        self._check_empty()
        self._check_columns()
        self._check_data_types()
        self._check_missing_values()
        self._check_duplicates()
        self._detect_datetime_columns()
        
        is_valid = len(self.issues) == 0
        return is_valid, self.issues, self.warnings
    
    def _check_empty(self):
        """Check if dataframe is empty"""
        if self.df.empty:
            self.issues.append("Dataset is empty")
        elif len(self.df) < 5:
            self.warnings.append(f"Dataset has only {len(self.df)} rows - may be too small for meaningful analysis")
    
    def _check_columns(self):
        """Check column count and names"""
        if len(self.df.columns) == 0:
            self.issues.append("No columns found in dataset")
        elif len(self.df.columns) < 2:
            self.warnings.append("Dataset has only 1 column - limited analysis possible")
        
        # Check for unnamed columns
        unnamed_cols = [col for col in self.df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            self.warnings.append(f"Found {len(unnamed_cols)} unnamed columns: {unnamed_cols[:3]}")
    
    def _check_data_types(self):
        """Analyze data types"""
        numeric_cols = self.df.select_dtypes(include='number').columns
        
        if len(numeric_cols) == 0:
            self.warnings.append("No numeric columns found - some visualizations may not be available")
    
    def _check_missing_values(self):
        """Check for missing values"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            self.warnings.append(
                f"{len(high_missing)} columns have >50% missing values: "
                f"{list(high_missing.index[:3])}"
            )
        
        total_missing = missing.sum()
        if total_missing > 0:
            self.warnings.append(f"Dataset contains {total_missing} missing values")
    
    def _check_duplicates(self):
        """Check for duplicate rows"""
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.warnings.append(f"Found {duplicates} duplicate rows ({duplicates/len(self.df)*100:.1f}%)")
    
    def _detect_datetime_columns(self):
        """Attempt to detect and convert datetime columns"""
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                try:
                    sample = self.df[col].dropna().head(100)
                    parsed = pd.to_datetime(sample, errors='coerce')
                    
                    if parsed.notna().sum() / len(sample) > 0.8:
                        self.warnings.append(f"Column '{col}' appears to be datetime but is stored as text")
                except:
                    pass
    
    def get_summary(self) -> Dict:
        """Get dataset summary statistics"""
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "numeric_columns": len(self.df.select_dtypes(include='number').columns),
            "categorical_columns": len(self.df.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": len(self.df.select_dtypes(include='datetime64').columns),
            "total_missing": self.df.isnull().sum().sum(),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / (1024**2)
        }
    
    def clean_data(self) -> pd.DataFrame:
        """
        Apply basic cleaning operations
        
        Returns:
            Cleaned dataframe
        """
        df_clean = self.df.copy()
        
        # Remove columns that are all NaN
        df_clean = df_clean.dropna(axis=1, how='all')
        
        # Remove duplicate rows
        df_clean = df_clean.drop_duplicates()
        
        # Strip whitespace from string columns
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].str.strip()
        
        # Convert obvious datetime columns
        for col in df_clean.select_dtypes(include=['object']).columns:
            try:
                parsed = pd.to_datetime(df_clean[col], errors='coerce')
                if parsed.notna().sum() / len(df_clean) > 0.9:
                    df_clean[col] = parsed
            except:
                pass
        
        return df_clean
