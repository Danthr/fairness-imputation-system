"""
KNN Imputation Engine
Implements K-Nearest Neighbors imputation with baseline comparison
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImputationEngine:
    """Handles missing value imputation using various strategies"""
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform'):
        """
        Initialize imputation engine
        
        Args:
            n_neighbors: Number of neighbors for KNN imputation
            weights: Weight function ('uniform' or 'distance')
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.knn_imputer = None
        self.label_encoders = {}
        self.categorical_columns = []
        self.numerical_columns = []
        self.imputation_stats = {}
    
    def _identify_column_types(self, df: pd.DataFrame) -> None:
        """Identify numerical and categorical columns"""
        self.numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Identified {len(self.numerical_columns)} numerical and {len(self.categorical_columns)} categorical columns")
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables for KNN imputation
        
        Returns:
            Encoded dataframe
        """
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            le = LabelEncoder()
            # Handle NaN values
            mask = df_encoded[col].notna()
            # Create a new array for encoded values
            encoded_values = np.full(len(df_encoded), np.nan)
            encoded_values[mask] = le.fit_transform(df_encoded.loc[mask, col])
            # Replace the column with numeric dtype
            df_encoded[col] = encoded_values
            self.label_encoders[col] = le
        
        return df_encoded
    
    def _decode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode categorical variables back to original values
        
        Returns:
            Decoded dataframe
        """
        df_decoded = df.copy()
        
        for col in self.categorical_columns:
            if col in self.label_encoders:
                # Round to nearest integer (in case of averaging in KNN)
                df_decoded[col] = df_decoded[col].round()
                # Decode
                le = self.label_encoders[col]
                # Ensure values are within valid range
                df_decoded[col] = df_decoded[col].clip(0, len(le.classes_) - 1)
                df_decoded[col] = le.inverse_transform(df_decoded[col].astype(int))
        
        return df_decoded
    
    def knn_impute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform KNN imputation on dataset
        
        Args:
            df: Input dataframe with missing values
            
        Returns:
            Tuple of (imputed dataframe, statistics dictionary)
        """
        logger.info("Starting KNN imputation...")
        start_time = time.time()
        
        # Identify column types
        self._identify_column_types(df)
        
        # Store original missing value counts
        missing_before = df.isnull().sum().sum()
        
        # Encode categorical variables
        df_encoded = self._encode_categorical(df)
        
        # Initialize KNN imputer
        self.knn_imputer = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights
        )
        
        # Perform imputation
        df_imputed = pd.DataFrame(
            self.knn_imputer.fit_transform(df_encoded),
            columns=df.columns,
            index=df.index
        )
        
        # Decode categorical variables
        df_imputed = self._decode_categorical(df_imputed)
        
        # Calculate statistics
        missing_after = df_imputed.isnull().sum().sum()
        execution_time = time.time() - start_time
        
        stats = {
            'method': 'KNN',
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'values_imputed': int(missing_before - missing_after),
            'execution_time_seconds': round(execution_time, 2)
        }
        
        self.imputation_stats['knn'] = stats
        logger.info(f"KNN imputation completed in {execution_time:.2f} seconds")
        
        return df_imputed, stats
    
    def mean_median_impute(self, df: pd.DataFrame, strategy: str = 'mean') -> Tuple[pd.DataFrame, Dict]:
        """
        Baseline imputation using mean/median for numerical and mode for categorical
        
        Args:
            df: Input dataframe
            strategy: 'mean' or 'median' for numerical columns
            
        Returns:
            Tuple of (imputed dataframe, statistics dictionary)
        """
        logger.info(f"Starting {strategy} imputation...")
        start_time = time.time()
        
        df_imputed = df.copy()
        missing_before = df.isnull().sum().sum()
        
        # Identify column types
        self._identify_column_types(df)
        
        # Impute numerical columns
        if self.numerical_columns:
            num_imputer = SimpleImputer(strategy=strategy)
            df_imputed[self.numerical_columns] = num_imputer.fit_transform(df[self.numerical_columns])
        
        # Impute categorical columns with mode
        if self.categorical_columns:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_imputed[self.categorical_columns] = cat_imputer.fit_transform(df[self.categorical_columns])
        
        missing_after = df_imputed.isnull().sum().sum()
        execution_time = time.time() - start_time
        
        stats = {
            'method': strategy.capitalize(),
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'values_imputed': int(missing_before - missing_after),
            'execution_time_seconds': round(execution_time, 2)
        }
        
        self.imputation_stats[strategy] = stats
        logger.info(f"{strategy.capitalize()} imputation completed in {execution_time:.2f} seconds")
        
        return df_imputed, stats
    
    def compare_methods(self, df: pd.DataFrame) -> Dict:
        """
        Compare KNN imputation against baseline methods
        
        Args:
            df: Input dataframe with missing values
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info("Comparing imputation methods...")
        
        results = {
            'original_missing_count': int(df.isnull().sum().sum()),
            'original_missing_percentage': round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
            'methods': {}
        }
        
        # KNN Imputation
        df_knn, knn_stats = self.knn_impute(df)
        results['methods']['knn'] = knn_stats
        
        # Mean Imputation
        df_mean, mean_stats = self.mean_median_impute(df, strategy='mean')
        results['methods']['mean'] = mean_stats
        
        # Median Imputation
        df_median, median_stats = self.mean_median_impute(df, strategy='median')
        results['methods']['median'] = median_stats
        
        # Determine recommended method (KNN is default for mixed data)
        results['recommended_method'] = 'knn'
        results['recommendation_reason'] = 'KNN works best for datasets with mixed numerical and categorical data'
        
        return results
    
    def get_imputation_report(self, df_original: pd.DataFrame, df_imputed: pd.DataFrame) -> pd.DataFrame:
        """
        Generate detailed imputation report per column
        
        Returns:
            DataFrame with imputation statistics per column
        """
        report_data = []
        
        for col in df_original.columns:
            missing_original = df_original[col].isnull().sum()
            missing_imputed = df_imputed[col].isnull().sum()
            
            report_data.append({
                'column': col,
                'missing_before': int(missing_original),
                'missing_after': int(missing_imputed),
                'values_imputed': int(missing_original - missing_imputed),
                'imputation_rate': round(((missing_original - missing_imputed) / missing_original * 100) if missing_original > 0 else 0, 2)
            })
        
        return pd.DataFrame(report_data)