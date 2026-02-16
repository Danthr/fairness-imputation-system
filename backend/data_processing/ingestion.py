"""
Data Ingestion Module
Loads, preprocesses, and analyzes datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging
from .validator import DataValidator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data loading and preprocessing"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.validator = DataValidator()
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, file_path: str) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Load dataset from file
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Tuple[DataFrame, message]: Loaded dataset and status message
        """
        try:
            # Validate file type
            if not self.validator.validate_file_type(file_path):
                return None, f"Unsupported file type. Supported: {self.validator.allowed_extensions}"
            
            # Load based on file extension
            file_extension = file_path[file_path.rfind('.'):].lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                return None, f"Unsupported file type: {file_extension}"
            
            logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Validate the dataframe
            is_valid, message = self.validator.validate_dataframe(df)
            if not is_valid:
                return None, message
            
            return df, "Dataset loaded successfully"
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return None, f"Error loading dataset: {str(e)}"
    
    def get_basic_stats(self, df: pd.DataFrame) -> Dict:
        """
        Generate basic statistics about the dataset
        
        Returns:
            Dictionary containing dataset statistics
        """
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        # Identify column types
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'total_cells': total_cells,
            'missing_cells': int(missing_cells),
            'missing_percentage': round((missing_cells / total_cells) * 100, 2),
            'numerical_columns': numerical_cols,
            'categorical_columns': categorical_cols,
            'protected_attributes': self.validator.detect_protected_attributes(df),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        return stats
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Initial preprocessing:
        - Remove duplicate rows
        - Strip whitespace from strings
        - Standardize column names
        
        Returns:
            Preprocessed dataframe
        """
        df_processed = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_processed)
        df_processed = df_processed.drop_duplicates()
        duplicates_removed = initial_rows - len(df_processed)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Strip whitespace from string columns
        string_cols = df_processed.select_dtypes(include=['object']).columns
        for col in string_cols:
            df_processed[col] = df_processed[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
        
        # Standardize column names (lowercase, replace spaces with underscores)
        df_processed.columns = df_processed.columns.str.lower().str.replace(' ', '_')
        
        logger.info("Preprocessing completed")
        
        return df_processed
    
    def save_dataset(self, df: pd.DataFrame, filename: str, directory: str = "processed") -> str:
        """
        Save dataset to file
        
        Args:
            df: DataFrame to save
            filename: Name of output file
            directory: 'raw' or 'processed'
            
        Returns:
            Path to saved file
        """
        if directory == "processed":
            save_dir = self.processed_dir
        else:
            save_dir = self.raw_dir
        
        file_path = save_dir / filename
        
        # Save based on extension
        if filename.endswith('.csv'):
            df.to_csv(file_path, index=False)
        elif filename.endswith('.xlsx'):
            df.to_excel(file_path, index=False)
        else:
            df.to_csv(file_path, index=False)
        
        logger.info(f"Dataset saved to {file_path}")
        return str(file_path)