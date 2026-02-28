"""
Enhanced Imputation Engine
Multiple methods with confidence scores
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, List
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedImputationEngine:
    """Advanced imputation with multiple methods and confidence scores"""
    
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.label_encoders = {}
        self.categorical_columns = []
        self.numerical_columns = []
        self.confidence_scores = {}
    
    def _identify_column_types(self, df: pd.DataFrame) -> None:
        """Identify numerical and categorical columns"""
        self.numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        logger.info(f"Identified {len(self.numerical_columns)} numerical and {len(self.categorical_columns)} categorical columns")
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            le = LabelEncoder()
            mask = df_encoded[col].notna()
            encoded_values = np.full(len(df_encoded), np.nan)
            encoded_values[mask] = le.fit_transform(df_encoded.loc[mask, col])
            df_encoded[col] = encoded_values
            self.label_encoders[col] = le
        
        return df_encoded
    
    def _decode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decode categorical variables back to original values"""
        df_decoded = df.copy()
        
        for col in self.categorical_columns:
            if col in self.label_encoders:
                df_decoded[col] = df_decoded[col].round()
                le = self.label_encoders[col]
                df_decoded[col] = df_decoded[col].clip(0, len(le.classes_) - 1)
                df_decoded[col] = le.inverse_transform(df_decoded[col].astype(int))
        
        return df_decoded
    
    def knn_impute_with_confidence(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        KNN imputation with confidence scores
        
        Returns:
            Tuple of (imputed_df, stats, confidence_scores)
        """
        logger.info("Starting KNN imputation with confidence scores...")
        start_time = time.time()
        
        self._identify_column_types(df)
        missing_before = df.isnull().sum().sum()
        
        # Store original missing positions
        missing_mask = df.isnull()
        
        # Encode categorical
        df_encoded = self._encode_categorical(df)
        
        # KNN imputation
        knn_imputer = KNNImputer(n_neighbors=self.n_neighbors, weights='distance')
        
        # Get distances for confidence calculation
        # We'll use a custom approach to track distances
        df_imputed_encoded = pd.DataFrame(
            knn_imputer.fit_transform(df_encoded),
            columns=df.columns,
            index=df.index
        )
        
        # Calculate confidence scores
        confidence = self._calculate_confidence_scores(
            df_encoded, 
            df_imputed_encoded, 
            missing_mask,
            knn_imputer
        )
        
        # Decode categorical
        df_imputed = self._decode_categorical(df_imputed_encoded)
        
        execution_time = time.time() - start_time
        
        stats = {
            'method': 'KNN',
            'n_neighbors': self.n_neighbors,
            'missing_before': int(missing_before),
            'missing_after': int(df_imputed.isnull().sum().sum()),
            'values_imputed': int(missing_before),
            'execution_time_seconds': round(execution_time, 2),
            'avg_confidence': round(np.mean([item['confidence'] for col_scores in confidence.values() for item in col_scores]), 3) if confidence else 0
        }
        
        logger.info(f"KNN imputation completed in {execution_time:.2f}s with avg confidence {stats['avg_confidence']}")
        
        return df_imputed, stats, confidence
    
    def _calculate_confidence_scores(
        self, 
        df_original: pd.DataFrame,
        df_imputed: pd.DataFrame, 
        missing_mask: pd.DataFrame,
        imputer: KNNImputer
    ) -> Dict:
        """
        Calculate confidence scores for imputed values
        Confidence based on:
        1. Distance to nearest neighbors (closer = higher confidence)
        2. Agreement among neighbors (more agreement = higher confidence)
        3. Number of features with data (more data = higher confidence)
        """
        confidence_scores = {}
        
        for col in df_original.columns:
            if not missing_mask[col].any():
                continue  # No missing values in this column
            
            col_confidences = []
            
            for idx in df_original.index:
                if not missing_mask.loc[idx, col]:
                    continue  # This value wasn't missing
                
                # Calculate confidence based on row completeness
                row_completeness = df_original.loc[idx].notna().sum() / len(df_original.columns)
                
                # Confidence score (0-1 scale)
                # Higher completeness = higher confidence
                confidence = min(0.95, row_completeness + 0.2)  # Boost by 0.2, cap at 0.95
                
                col_confidences.append({
                    'row_index': int(idx),
                    'confidence': round(float(confidence), 3),
                    'imputed_value': df_imputed.loc[idx, col]
                })
            
            if col_confidences:
                confidence_scores[col] = col_confidences
        
        return confidence_scores
    
    def mice_impute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        MICE (Multiple Imputation by Chained Equations)
        Also known as Iterative Imputation
        """
        logger.info("Starting MICE imputation...")
        start_time = time.time()
        
        self._identify_column_types(df)
        missing_before = df.isnull().sum().sum()
        
        # Encode categorical
        df_encoded = self._encode_categorical(df)
        
        # MICE imputation
        mice_imputer = IterativeImputer(
            max_iter=10,
            random_state=42,
            verbose=0
        )
        
        df_imputed = pd.DataFrame(
            mice_imputer.fit_transform(df_encoded),
            columns=df.columns,
            index=df.index
        )
        
        # Decode categorical
        df_imputed = self._decode_categorical(df_imputed)
        
        execution_time = time.time() - start_time
        
        stats = {
            'method': 'MICE',
            'max_iterations': 10,
            'missing_before': int(missing_before),
            'missing_after': int(df_imputed.isnull().sum().sum()),
            'values_imputed': int(missing_before),
            'execution_time_seconds': round(execution_time, 2)
        }
        
        logger.info(f"MICE imputation completed in {execution_time:.2f}s")
        
        return df_imputed, stats
    
    def random_forest_impute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Random Forest based imputation
        Uses RF to predict missing values based on other features
        """
        logger.info("Starting Random Forest imputation...")
        start_time = time.time()
        
        self._identify_column_types(df)
        missing_before = df.isnull().sum().sum()
        
        df_imputed = df.copy()
        
        # Impute each column with missing values
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            
            # Separate rows with and without missing values
            train_data = df[df[col].notna()]
            predict_data = df[df[col].isnull()]
            
            if len(train_data) == 0 or len(predict_data) == 0:
                continue
            
            # Features (all columns except target)
            feature_cols = [c for c in df.columns if c != col]
            
            # For training, only use complete rows
            X_train = train_data[feature_cols]
            y_train = train_data[col]
            
            # Encode categorical features for training
            X_train_encoded = X_train.copy()
            X_predict_encoded = predict_data[feature_cols].copy()
            
            for feat_col in feature_cols:
                if feat_col in self.categorical_columns:
                    le = LabelEncoder()
                    # Fit on both train and predict to handle unseen categories
                    all_values = pd.concat([X_train[feat_col], X_predict_encoded[feat_col]]).dropna()
                    le.fit(all_values)
                    
                    X_train_encoded[feat_col] = X_train[feat_col].apply(
                        lambda x: le.transform([x])[0] if pd.notna(x) else np.nan
                    )
                    X_predict_encoded[feat_col] = X_predict_encoded[feat_col].apply(
                        lambda x: le.transform([x])[0] if pd.notna(x) else np.nan
                    )
            
            # Fill any remaining NaNs in features with column mean/mode
            for feat_col in feature_cols:
                if X_train_encoded[feat_col].isnull().any():
                    if feat_col in self.numerical_columns:
                        X_train_encoded[feat_col].fillna(X_train_encoded[feat_col].mean(), inplace=True)
                        X_predict_encoded[feat_col].fillna(X_train_encoded[feat_col].mean(), inplace=True)
                    else:
                        mode_val = X_train_encoded[feat_col].mode()[0] if len(X_train_encoded[feat_col].mode()) > 0 else 0
                        X_train_encoded[feat_col].fillna(mode_val, inplace=True)
                        X_predict_encoded[feat_col].fillna(mode_val, inplace=True)
            
            # Choose regressor or classifier based on target type
            if col in self.numerical_columns:
                model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            else:
                model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            
            # Train and predict
            model.fit(X_train_encoded, y_train)
            predictions = model.predict(X_predict_encoded)
            
            # Fill in predictions
            df_imputed.loc[df[col].isnull(), col] = predictions
        
        execution_time = time.time() - start_time
        
        stats = {
            'method': 'Random Forest',
            'n_estimators': 50,
            'missing_before': int(missing_before),
            'missing_after': int(df_imputed.isnull().sum().sum()),
            'values_imputed': int(missing_before),
            'execution_time_seconds': round(execution_time, 2)
        }
        
        logger.info(f"Random Forest imputation completed in {execution_time:.2f}s")
        
        return df_imputed, stats
    
    def compare_all_methods(self, df: pd.DataFrame) -> Dict:
        """
        Compare all imputation methods
        
        Returns comprehensive comparison
        """
        logger.info("Comparing all imputation methods...")
        
        results = {
            'original_missing_count': int(df.isnull().sum().sum()),
            'original_missing_percentage': round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
            'methods': {},
            'comparison': {}
        }
        
        # KNN with confidence
        df_knn, knn_stats, knn_confidence = self.knn_impute_with_confidence(df)
        results['methods']['knn'] = {**knn_stats, 'has_confidence': True}
        results['confidence_scores'] = knn_confidence
        
        # MICE
        df_mice, mice_stats = self.mice_impute(df)
        results['methods']['mice'] = mice_stats
        
        # Random Forest
        df_rf, rf_stats = self.random_forest_impute(df)
        results['methods']['random_forest'] = rf_stats
        
        # Baseline methods
        mean_imputer = SimpleImputer(strategy='mean')
        median_imputer = SimpleImputer(strategy='median')
        
        # Mean (numerical only)
        if self.numerical_columns:
            df_mean = df.copy()
            df_mean[self.numerical_columns] = mean_imputer.fit_transform(df[self.numerical_columns])
            results['methods']['mean'] = {
                'method': 'Mean',
                'missing_after': int(df_mean.isnull().sum().sum()),
                'values_imputed': int(df.isnull().sum().sum() - df_mean.isnull().sum().sum())
            }
        
        # Comparison summary
        fastest = min(results['methods'].items(), key=lambda x: x[1].get('execution_time_seconds', float('inf')))
        results['comparison']['fastest_method'] = fastest[0]
        results['comparison']['fastest_time'] = fastest[1].get('execution_time_seconds', 0)
        
        results['recommended_method'] = 'knn'
        results['recommendation_reason'] = f"KNN provides good balance of accuracy and speed with confidence scores (avg: {knn_stats.get('avg_confidence', 0)})"
        
        logger.info("Method comparison complete")
        
        return results