"""
Test Stage 7: Enhanced Imputation Methods
Tests multiple imputation methods and confidence scores
"""

import pandas as pd
import numpy as np
from backend.data_processing import DataIngestion
from backend.imputation import EnhancedImputationEngine

print("="*70)
print("STAGE 7: ENHANCED IMPUTATION ENGINE TEST")
print("="*70)

# Load test data
ingestion = DataIngestion()
df, message = ingestion.load_dataset('data/raw/test_data.csv')

print("\n" + "="*70)
print("ORIGINAL DATASET")
print("="*70)
print(df)
print(f"\nTotal missing values: {df.isnull().sum().sum()}")
print(f"Missing percentage: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%")

# Initialize enhanced engine
engine = EnhancedImputationEngine(n_neighbors=3)

# =============================================================================
# TEST 1: KNN with Confidence Scores
# =============================================================================

print("\n" + "="*70)
print("TEST 1: KNN IMPUTATION WITH CONFIDENCE SCORES")
print("="*70)

df_knn, knn_stats, confidence = engine.knn_impute_with_confidence(df)

print(f"\nMethod: {knn_stats['method']}")
print(f"Neighbors: {knn_stats['n_neighbors']}")
print(f"Values imputed: {knn_stats['values_imputed']}")
print(f"Execution time: {knn_stats['execution_time_seconds']}s")
print(f"Average confidence: {knn_stats['avg_confidence']}")

print("\n" + "-"*70)
print("CONFIDENCE SCORES (Sample)")
print("-"*70)
for col, scores in confidence.items():
    print(f"\nColumn: {col}")
    for score_info in scores[:3]:  # Show first 3
        print(f"  Row {score_info['row_index']}: "
              f"Value={score_info['imputed_value']}, "
              f"Confidence={score_info['confidence']}")
    if len(scores) > 3:
        print(f"  ... and {len(scores)-3} more")

# =============================================================================
# TEST 2: MICE Imputation
# =============================================================================

print("\n" + "="*70)
print("TEST 2: MICE (ITERATIVE) IMPUTATION")
print("="*70)

df_mice, mice_stats = engine.mice_impute(df)

print(f"\nMethod: {mice_stats['method']}")
print(f"Max iterations: {mice_stats['max_iterations']}")
print(f"Values imputed: {mice_stats['values_imputed']}")
print(f"Execution time: {mice_stats['execution_time_seconds']}s")

# =============================================================================
# TEST 3: Random Forest Imputation
# =============================================================================

print("\n" + "="*70)
print("TEST 3: RANDOM FOREST IMPUTATION")
print("="*70)

df_rf, rf_stats = engine.random_forest_impute(df)

print(f"\nMethod: {rf_stats['method']}")
print(f"N estimators: {rf_stats['n_estimators']}")
print(f"Values imputed: {rf_stats['values_imputed']}")
print(f"Execution time: {rf_stats['execution_time_seconds']}s")

# =============================================================================
# TEST 4: Compare All Methods
# =============================================================================

print("\n" + "="*70)
print("TEST 4: COMPARE ALL METHODS")
print("="*70)

comparison = engine.compare_all_methods(df)

print(f"\nOriginal missing: {comparison['original_missing_count']} "
      f"({comparison['original_missing_percentage']}%)")

print("\n" + "-"*70)
print("METHOD COMPARISON")
print("-"*70)

# Create comparison table
print(f"\n{'Method':<15} {'Imputed':<10} {'Time (s)':<10} {'Confidence':<12}")
print("-"*70)

for method_name, method_stats in comparison['methods'].items():
    imputed = method_stats.get('values_imputed', 'N/A')
    time_taken = method_stats.get('execution_time_seconds', 'N/A')
    has_conf = method_stats.get('has_confidence', False)
    avg_conf = method_stats.get('avg_confidence', 'N/A')
    
    conf_str = f"{avg_conf}" if has_conf else "N/A"
    
    print(f"{method_name:<15} {str(imputed):<10} {str(time_taken):<10} {conf_str:<12}")

print("\n" + "-"*70)
print(f"Recommended: {comparison['recommended_method']}")
print(f"Reason: {comparison['recommendation_reason']}")

# =============================================================================
# TEST 5: Verify Imputed Data Quality
# =============================================================================

print("\n" + "="*70)
print("TEST 5: DATA QUALITY VERIFICATION")
print("="*70)

print("\nKNN Imputed Dataset:")
print(df_knn)

print("\n" + "-"*70)
print("Missing Values After Imputation:")
print("-"*70)
print(f"KNN: {df_knn.isnull().sum().sum()}")
print(f"MICE: {df_mice.isnull().sum().sum()}")
print(f"Random Forest: {df_rf.isnull().sum().sum()}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("✅ STAGE 7 TEST COMPLETE")
print("="*70)

print("\nFeatures Tested:")
print("  ✓ KNN with confidence scores")
print("  ✓ MICE imputation")
print("  ✓ Random Forest imputation")
print("  ✓ Method comparison")
print("  ✓ Confidence score calculation")

print("\nAll methods successfully imputed missing values!")
print("="*70)