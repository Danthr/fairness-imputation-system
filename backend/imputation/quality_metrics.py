"""
Imputation Quality Metrics - Stage 7 Feature #4
Evaluates imputation accuracy using RMSE, MAE, and cross-validation.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImputationQualityMetrics:
    """
    Evaluates imputation quality by:
    1. Taking a complete (or partially complete) dataset
    2. Artificially masking some known values
    3. Imputing those masked values
    4. Comparing imputed vs actual using RMSE, MAE, CV
    """

    def __init__(self, mask_fraction: float = 0.2, random_state: int = 42):
        """
        Args:
            mask_fraction: Fraction of known values to mask for evaluation (default 20%)
            random_state:  Seed for reproducibility
        """
        self.mask_fraction = mask_fraction
        self.random_state = random_state
        self.label_encoders: Dict[str, LabelEncoder] = {}

    # ------------------------------------------------------------------
    # Core metric calculations
    # ------------------------------------------------------------------

    def calculate_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Root Mean Square Error - penalises large errors more."""
        actual    = np.array(actual,    dtype=float)
        predicted = np.array(predicted, dtype=float)
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        if mask.sum() == 0:
            return float('nan')
        return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))

    def calculate_mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Mean Absolute Error - average magnitude of errors."""
        actual    = np.array(actual,    dtype=float)
        predicted = np.array(predicted, dtype=float)
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        if mask.sum() == 0:
            return float('nan')
        return float(np.mean(np.abs(actual[mask] - predicted[mask])))

    def calculate_categorical_accuracy(
        self, actual: np.ndarray, predicted: np.ndarray
    ) -> float:
        """Exact-match accuracy for categorical columns."""
        actual    = np.array(actual)
        predicted = np.array(predicted)
        mask = (actual != None) & (predicted != None)  # noqa: E711
        mask = mask & pd.notnull(actual) & pd.notnull(predicted)
        if mask.sum() == 0:
            return float('nan')
        return float(np.mean(actual[mask] == predicted[mask]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """Label-encode all object columns. Returns encoded df + encoders."""
        df_enc = df.copy()
        encoders: Dict[str, LabelEncoder] = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            le.fit(df_enc[col].dropna())
            known_mask = df_enc[col].notna()
            encoded = np.full(len(df_enc), np.nan)
            encoded[known_mask] = le.transform(df_enc.loc[known_mask, col])
            df_enc[col] = encoded
            encoders[col] = le
        return df_enc, encoders

    def _impute_array(
        self, arr: np.ndarray, method: str, n_neighbors: int = 5
    ) -> np.ndarray:
        """Run the chosen sklearn imputer on a 2D numpy array."""
        if method == 'knn':
            imp = KNNImputer(n_neighbors=n_neighbors)
        elif method == 'mice':
            imp = IterativeImputer(max_iter=10, random_state=self.random_state)
        elif method == 'mean':
            imp = SimpleImputer(strategy='mean')
        elif method == 'rf':
            imp = IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=50, random_state=self.random_state
                ),
                max_iter=5,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown method: '{method}'. Choose knn/mice/rf/mean.")
        return imp.fit_transform(arr)

    def _mask_known_values(
        self, df_enc: pd.DataFrame, numerical_cols: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Artificially remove mask_fraction of known values from numerical columns.
        Returns the masked df and a dict of {col: array_of_masked_indices}.
        """
        rng = np.random.default_rng(self.random_state)
        df_masked = df_enc.copy()
        masked_indices: Dict[str, np.ndarray] = {}

        for col in numerical_cols:
            known_idx = df_masked[col].dropna().index.tolist()
            if len(known_idx) == 0:
                continue
            n_mask = max(1, int(len(known_idx) * self.mask_fraction))
            chosen = rng.choice(known_idx, size=n_mask, replace=False)
            masked_indices[col] = chosen
            df_masked.loc[chosen, col] = np.nan

        return df_masked, masked_indices

    # ------------------------------------------------------------------
    # Single-method evaluation
    # ------------------------------------------------------------------

    def evaluate_method(
        self, df: pd.DataFrame, method: str = 'knn', n_neighbors: int = 5
    ) -> Dict:
        """
        Evaluate one imputation method on a dataset.

        Process:
          1. Encode categoricals
          2. Mask 20% of known numerical values
          3. Impute with chosen method
          4. Compare imputed vs actual → RMSE, MAE per column + overall

        Returns a dict with rmse, mae, per_column breakdown, and timing.
        """
        start = time.time()

        df_enc, encoders = self._encode_df(df)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Also treat encoded categoricals as numerical for imputation
        all_num_cols = df_enc.select_dtypes(include=[np.number]).columns.tolist()

        if len(all_num_cols) == 0:
            return {'error': 'No numerical columns found for quality evaluation'}

        # Step 1: mask some known values
        df_masked, masked_indices = self._mask_known_values(df_enc, all_num_cols)

        if not masked_indices:
            return {'error': 'Not enough known values to mask for evaluation'}

        # Step 2: impute
        arr_masked   = df_masked[all_num_cols].values.astype(float)
        arr_imputed  = self._impute_array(arr_masked, method, n_neighbors)
        df_imputed   = pd.DataFrame(arr_imputed, columns=all_num_cols, index=df_enc.index)

        # Step 3: collect actual vs predicted for masked positions only
        all_actual    = []
        all_predicted = []
        per_column: Dict[str, Dict] = {}

        for col, idx in masked_indices.items():
            actual_vals    = df_enc.loc[idx, col].values.astype(float)
            predicted_vals = df_imputed.loc[idx, col].values.astype(float)

            col_rmse = self.calculate_rmse(actual_vals, predicted_vals)
            col_mae  = self.calculate_mae(actual_vals, predicted_vals)

            per_column[col] = {
                'rmse':          round(col_rmse, 4),
                'mae':           round(col_mae,  4),
                'n_evaluated':   len(idx),
            }

            all_actual.extend(actual_vals.tolist())
            all_predicted.extend(predicted_vals.tolist())

        overall_rmse = self.calculate_rmse(
            np.array(all_actual), np.array(all_predicted)
        )
        overall_mae = self.calculate_mae(
            np.array(all_actual), np.array(all_predicted)
        )

        elapsed = round(time.time() - start, 4)

        return {
            'method':       method,
            'rmse':         round(overall_rmse, 4),
            'mae':          round(overall_mae,  4),
            'per_column':   per_column,
            'n_evaluated':  len(all_actual),
            'eval_time_s':  elapsed,
        }

    # ------------------------------------------------------------------
    # Cross-validation evaluation
    # ------------------------------------------------------------------

    def cross_validate_method(
        self,
        df: pd.DataFrame,
        method: str = 'knn',
        n_folds: int = 5,
        n_neighbors: int = 5,
    ) -> Dict:
        """
        K-Fold cross-validation for imputation quality.

        For each fold:
          - Treat that fold's known values as the "test" set (mask them)
          - Impute using remaining data
          - Calculate RMSE and MAE

        Returns mean and std of RMSE/MAE across all folds.
        """
        df_enc, _ = self._encode_df(df)
        numerical_cols = df_enc.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) == 0:
            return {'error': 'No numerical columns for cross-validation'}

        # Collect all known positions across numerical columns
        known_positions = []
        for col in numerical_cols:
            for idx in df_enc[col].dropna().index:
                known_positions.append((idx, col))

        if len(known_positions) < n_folds:
            # Not enough data — fall back to 2 folds
            n_folds = max(2, len(known_positions) // 2)
            logger.warning(f"Not enough known values, reducing to {n_folds} folds")

        kf = __import__('sklearn.model_selection', fromlist=['KFold']).KFold(
            n_splits=n_folds, shuffle=True, random_state=self.random_state
        )

        fold_rmse_list = []
        fold_mae_list  = []

        positions_arr = np.array(known_positions, dtype=object)

        for fold_idx, (train_pos, test_pos) in enumerate(kf.split(positions_arr)):
            df_fold = df_enc.copy()
            test_positions = positions_arr[test_pos]

            # Mask test positions
            actual_vals    = []
            predicted_vals = []

            for row_idx, col in test_positions:
                actual_vals.append(float(df_fold.loc[row_idx, col]))
                df_fold.loc[row_idx, col] = np.nan

            # Impute
            arr   = df_fold[numerical_cols].values.astype(float)
            arr_i = self._impute_array(arr, method, n_neighbors)
            df_i  = pd.DataFrame(arr_i, columns=numerical_cols, index=df_enc.index)

            for row_idx, col in test_positions:
                predicted_vals.append(float(df_i.loc[row_idx, col]))

            fold_rmse = self.calculate_rmse(
                np.array(actual_vals), np.array(predicted_vals)
            )
            fold_mae = self.calculate_mae(
                np.array(actual_vals), np.array(predicted_vals)
            )
            fold_rmse_list.append(fold_rmse)
            fold_mae_list.append(fold_mae)

            logger.info(
                f"  Fold {fold_idx + 1}/{n_folds} — RMSE: {fold_rmse:.4f}, MAE: {fold_mae:.4f}"
            )

        return {
            'method':        method,
            'n_folds':       n_folds,
            'cv_rmse_mean':  round(float(np.mean(fold_rmse_list)),  4),
            'cv_rmse_std':   round(float(np.std(fold_rmse_list)),   4),
            'cv_mae_mean':   round(float(np.mean(fold_mae_list)),   4),
            'cv_mae_std':    round(float(np.std(fold_mae_list)),    4),
            'fold_scores':   [
                {
                    'fold':  i + 1,
                    'rmse':  round(r, 4),
                    'mae':   round(m, 4),
                }
                for i, (r, m) in enumerate(zip(fold_rmse_list, fold_mae_list))
            ],
        }

    # ------------------------------------------------------------------
    # Multi-method comparison
    # ------------------------------------------------------------------

    def compare_method_quality(
        self,
        df: pd.DataFrame,
        methods: List[str] = None,
        n_neighbors: int = 5,
        include_cv: bool = False,
    ) -> Dict:
        """
        Compare quality of multiple imputation methods on the same dataset.

        Args:
            df:          Input dataframe (may have missing values)
            methods:     List of methods to compare. Default: ['knn','mice','rf','mean']
            n_neighbors: For KNN
            include_cv:  If True, also run cross-validation for each method

        Returns a dict with per-method metrics and a ranking by RMSE.
        """
        if methods is None:
            methods = ['knn', 'mice', 'rf', 'mean']

        results: Dict[str, Dict] = {}

        for method in methods:
            logger.info(f"Evaluating method: {method}")
            eval_result = self.evaluate_method(df, method=method, n_neighbors=n_neighbors)

            if include_cv:
                cv_result = self.cross_validate_method(
                    df, method=method, n_neighbors=n_neighbors
                )
                eval_result['cross_validation'] = cv_result

            results[method] = eval_result

        # Rank methods by overall RMSE (lower = better)
        ranked = sorted(
            [m for m in methods if 'error' not in results[m]],
            key=lambda m: results[m].get('rmse', float('inf')),
        )

        best_method = ranked[0] if ranked else None

        return {
            'methods':      results,
            'ranking':      ranked,
            'best_method':  best_method,
            'summary': {
                m: {
                    'rmse': results[m].get('rmse'),
                    'mae':  results[m].get('mae'),
                }
                for m in methods if 'error' not in results[m]
            },
        }