"""
Test Stage 2: Data Ingestion Module
"""

import pandas as pd
import numpy as np
from backend.data_processing import DataIngestion, DataValidator

# Create sample dataset with missing values
print("Creating sample dataset...")
sample_data = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50, np.nan, 60, 35, 40, 55],
    'income': [50000, 60000, np.nan, 80000, np.nan, 75000, 95000, np.nan, 85000, 90000],
    'gender': ['M', 'F', 'M', np.nan, 'F', 'M', 'F', 'M', np.nan, 'F'],
    'education': ['BS', 'MS', 'PhD', 'BS', np.nan, 'MS', 'PhD', 'BS', 'MS', 'PhD']
})

print("\n" + "="*50)
print("ORIGINAL DATASET")
print("="*50)
print(sample_data)
print(f"\nMissing values: {sample_data.isnull().sum().sum()}")

# Initialize data ingestion
ingestion = DataIngestion()

# Save sample data
print("\n" + "="*50)
print("SAVING DATASET")
print("="*50)
sample_path = ingestion.save_dataset(sample_data, 'test_data.csv', 'raw')
print(f"Saved to: {sample_path}")

# Load dataset
print("\n" + "="*50)
print("LOADING DATASET")
print("="*50)
df, message = ingestion.load_dataset(sample_path)
if df is not None:
    print(f"✅ {message}")
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
else:
    print(f"❌ {message}")

# Get statistics
print("\n" + "="*50)
print("DATASET STATISTICS")
print("="*50)
stats = ingestion.get_basic_stats(df)
for key, value in stats.items():
    print(f"{key}: {value}")

# Get missing value summary
print("\n" + "="*50)
print("MISSING VALUE SUMMARY")
print("="*50)
validator = DataValidator()
missing_summary = validator.get_missing_value_summary(df)
print(missing_summary)

# Preprocess
print("\n" + "="*50)
print("PREPROCESSING")
print("="*50)
df_processed = ingestion.preprocess_dataset(df)
print(f"Processed dataset shape: {df_processed.shape}")
print(f"Column names: {list(df_processed.columns)}")

print("\n" + "="*50)
print("✅ STAGE 2 TEST COMPLETE")
print("="*50)