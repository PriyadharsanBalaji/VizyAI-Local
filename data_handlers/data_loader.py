import pandas as pd
from pathlib import Path

class DataLoader:
    """Handle various data formats"""
    
    @staticmethod
    def load_file(uploaded_file):
        """Load CSV or Excel file"""
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension == '.csv':
            return pd.read_csv(uploaded_file)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
