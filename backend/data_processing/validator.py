"""
Data Validator
Validates uploaded datasets for structure and quality
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class DataValidator:
    """Validates datasets before processing"""
    
    def __init__(self):
        self.min_rows = 10
        self.min_columns = 2
        self.max_missing_ratio = 0.9  # Allow up to 90% missing values
        self.allowed_extensions = ['.csv', '.xlsx', '.xls']
    
    def validate_file_type(self, file_path: str) -> bool:
        """Check if file type is supported"""
        extension = file_path[file_path.rfind('.'):].lower()
        return extension in self.allowed_extensions
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate dataframe structure and quality
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check minimum rows
        if len(df) < self.min_rows:
            return False, f"Dataset must have at least {self.min_rows} rows. Found: {len(df)}"
        
        # Check minimum columns
        if len(df.columns) < self.min_columns:
            return False, f"Dataset must have at least {self.min_columns} columns. Found: {len(df.columns)}"
        
        # Check if completely empty
        if df.empty:
            return False, "Dataset is completely empty"
        
        # Check missing value ratio
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells
        
        if missing_ratio > self.max_missing_ratio:
            return False, f"Too many missing values: {missing_ratio:.2%}"
        
        return True, "Dataset is valid"
    
    def detect_protected_attributes(self, df: pd.DataFrame) -> List[str]:
        """
        Detect potential protected attributes (gender, race, age, etc.)
        
        Returns:
            List of column names that might be protected attributes
        """
        protected_keywords = [
            'gender', 'sex', 'race', 'ethnicity', 'age', 
            'religion', 'disability', 'marital', 'nationality'
        ]
        
        protected_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in protected_keywords):
                protected_cols.append(col)
        
        return protected_cols
    
    def get_missing_value_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary of missing values per column
        
        Returns:
            DataFrame with missing value statistics
        """
        missing_info = []
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            missing_info.append({
                'column_name': col,
                'missing_count': int(missing_count),
                'missing_percentage': round(missing_pct, 2),
                'data_type': str(df[col].dtype)
            })
        
        return pd.DataFrame(missing_info)