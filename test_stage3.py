"""
Test Stage 3: KNN Imputation Engine
"""

import pandas as pd
import numpy as np
from backend.data_processing import DataIngestion
from backend.imputation import ImputationEngine

print("="*60)
print("STAGE 3: KNN IMPUTATION ENGINE TEST")
print("="*60)

# Load the test data we created in Stage 2
ingestion = DataIngestion()
df, message = ingestion.load_dataset('data/raw/test_data.csv')

print("\n" + "="*60)
print("ORIGINAL DATASET")
print("="*60)
print(df)
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# Initialize imputation engine with 3 neighbors
engine = ImputationEngine(n_neighbors=3)

# Compare all methods
print("\n" + "="*60)
print("COMPARING IMPUTATION METHODS")
print("="*60)
comparison = engine.compare_methods(df)

print(f"\nOriginal missing values: {comparison['original_missing_count']} ({comparison['original_missing_percentage']}%)")
print(f"\nRecommended method: {comparison['recommended_method']}")
print(f"Reason: {comparison['recommendation_reason']}")

print("\n" + "-"*60)
print("METHOD PERFORMANCE COMPARISON")
print("-"*60)
for method, stats in comparison['methods'].items():
    print(f"\n{method.upper()}:")
    print(f"  Values imputed: {stats['values_imputed']}")
    print(f"  Execution time: {stats['execution_time_seconds']}s")
    print(f"  Missing after: {stats['missing_after']}")

# Get imputed dataset using KNN
print("\n" + "="*60)
print("FINAL IMPUTED DATASET (KNN)")
print("="*60)
df_imputed, knn_stats = engine.knn_impute(df)
print(df_imputed)

# Generate imputation report
print("\n" + "="*60)
print("IMPUTATION REPORT (PER COLUMN)")
print("="*60)
report = engine.get_imputation_report(df, df_imputed)
print(report)

# Save imputed dataset
print("\n" + "="*60)
print("SAVING IMPUTED DATASET")
print("="*60)
output_path = ingestion.save_dataset(df_imputed, 'test_data_imputed.csv', 'processed')
print(f"Saved to: {output_path}")

print("\n" + "="*60)
print("✅ STAGE 3 TEST COMPLETE")
print("="*60)
