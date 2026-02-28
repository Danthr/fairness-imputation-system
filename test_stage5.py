"""
Test Stage 5: Flask REST API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5000"

print("="*60)
print("STAGE 5: FLASK API TEST")
print("="*60)

print("\nMake sure the API is running first!")
print("In another terminal, run: python backend/api/app.py")
print("\nPress Enter when the API is running...")
input()

# Test 1: Health check
print("\n" + "="*60)
print("TEST 1: Health Check")
print("="*60)
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
    print("Make sure API is running!")
    exit(1)

# Test 2: Upload file
print("\n" + "="*60)
print("TEST 2: Upload Dataset")
print("="*60)

# Use the test data we created earlier
file_path = "data/raw/test_data.csv"

try:
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/api/upload", files=files)
    
    print(f"Status Code: {response.status_code}")
    upload_result = response.json()
    print(json.dumps(upload_result, indent=2))
    
    dataset_id = upload_result.get('dataset_id')
    print(f"\n✅ Dataset uploaded with ID: {dataset_id}")
    
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Test 3: Process dataset
print("\n" + "="*60)
print("TEST 3: Process Dataset (Imputation + Fairness Audit)")
print("="*60)

try:
    payload = {
        'dataset_id': dataset_id,
        'n_neighbors': 3,
        'protected_attributes': ['gender', 'age']
    }
    
    response = requests.post(
        f"{BASE_URL}/api/process",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    process_result = response.json()
    
    print("\n--- Imputation Stats ---")
    print(json.dumps(process_result['imputation']['stats'], indent=2))
    
    print("\n--- Fairness Audit Results ---")
    for attr, result in process_result['fairness_audit'].items():
        if 'error' not in result:
            print(f"\n{attr.upper()}:")
            print(f"  Overall: {result['overall_assessment']['summary']}")
            print(f"  Metrics Passed: {result['overall_assessment']['metrics_passed']}/2")
    
    print(f"\n✅ Processing complete!")
    print(f"Output file: {process_result['output_file']}")
    
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Test 4: Get results
print("\n" + "="*60)
print("TEST 4: Get Results")
print("="*60)

try:
    response = requests.get(f"{BASE_URL}/api/results/{dataset_id}")
    
    print(f"Status Code: {response.status_code}")
    results = response.json()
    
    print(f"\nDataset: {results['filename']}")
    print(f"Status: {results['status']}")
    print(f"Total Rows: {results['stats']['total_rows']}")
    print(f"Missing Values: {results['stats']['missing_percentage']}%")
    print(f"Protected Attributes: {results['protected_attributes']}")
    
    print("\n✅ Results retrieved successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Test 5: List all datasets
print("\n" + "="*60)
print("TEST 5: List All Datasets")
print("="*60)

try:
    response = requests.get(f"{BASE_URL}/api/datasets")
    
    print(f"Status Code: {response.status_code}")
    datasets_list = response.json()
    
    print(f"\nTotal Datasets: {datasets_list['total_datasets']}")
    for dataset in datasets_list['datasets']:
        print(f"\n  ID: {dataset['dataset_id']}")
        print(f"  File: {dataset['filename']}")
        print(f"  Processed: {dataset['processed']}")
    
    print("\n✅ Dataset list retrieved!")
    
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("✅ STAGE 5 API TEST COMPLETE")
print("="*60)