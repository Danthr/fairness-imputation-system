# API Documentation

## Base URL
```
http://localhost:5000
```

---

## Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check API health |
| GET | `/` | API info |
| POST | `/api/upload` | Upload dataset |
| POST | `/api/process` | Process dataset (imputation + audit) |
| GET | `/api/results/{dataset_id}` | Get processing results |
| GET | `/api/datasets` | List all datasets |

---

## 1. Health Check

**Endpoint:** `GET /health`

**Description:** Check if API is running

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running"
}
```

---

## 2. Upload Dataset

**Endpoint:** `POST /api/upload`

**Description:** Upload a CSV or Excel file for processing

**Request:**
- Content-Type: `multipart/form-data`
- Body: Form data with file field named `file`

**Supported File Types:**
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)

**Example (Python):**
```python
import requests

url = "http://localhost:5000/api/upload"
files = {'file': open('data.csv', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

**Example (cURL):**
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@data.csv"
```

**Response:**
```json
{
  "dataset_id": "uuid-here",
  "filename": "data.csv",
  "message": "File uploaded successfully",
  "stats": {
    "total_rows": 100,
    "total_columns": 5,
    "missing_cells": 10,
    "missing_percentage": 2.0,
    "numerical_columns": ["age", "income"],
    "categorical_columns": ["gender", "education"],
    "protected_attributes": ["gender", "age"]
  }
}
```

---

## 3. Process Dataset

**Endpoint:** `POST /api/process`

**Description:** Run KNN imputation and fairness audit on uploaded dataset

**Request Body:**
```json
{
  "dataset_id": "uuid-from-upload",
  "n_neighbors": 5,  // optional, default: 5
  "protected_attributes": ["gender", "race"]  // optional, auto-detect if not provided
}
```

**Example (Python):**
```python
import requests

url = "http://localhost:5000/api/process"
payload = {
    "dataset_id": "your-dataset-id",
    "n_neighbors": 5,
    "protected_attributes": ["gender", "age"]
}
response = requests.post(url, json=payload)
print(response.json())
```

**Response:**
```json
{
  "dataset_id": "uuid",
  "message": "Processing complete",
  "imputation": {
    "stats": {
      "method": "KNN",
      "n_neighbors": 5,
      "missing_before": 10,
      "missing_after": 0,
      "values_imputed": 10,
      "execution_time_seconds": 0.05
    },
    "report": [
      {
        "column": "age",
        "missing_before": 5,
        "missing_after": 0,
        "values_imputed": 5,
        "imputation_rate": 100.0
      }
    ]
  },
  "fairness_audit": {
    "gender": {
      "protected_attribute": "gender",
      "overall_assessment": {
        "metrics_passed": 2,
        "total_metrics": 2,
        "is_fair": true,
        "summary": "Dataset passes fairness checks"
      },
      "metrics": {
        "disparate_impact": {
          "disparate_impact_score": 0.85,
          "is_fair": true,
          "interpretation": "Fair (passes 4/5ths rule)"
        }
      }
    }
  },
  "output_file": "path/to/processed.csv"
}
```

---

## 4. Get Results

**Endpoint:** `GET /api/results/{dataset_id}`

**Description:** Retrieve results for a processed dataset

**Example (Python):**
```python
import requests

url = "http://localhost:5000/api/results/your-dataset-id"
response = requests.get(url)
print(response.json())
```

**Response:**
```json
{
  "dataset_id": "uuid",
  "status": "processed",
  "filename": "data.csv",
  "stats": { ... },
  "imputation": { ... },
  "fairness_audit": { ... },
  "protected_attributes": ["gender", "age"],
  "output_file": "path/to/processed.csv"
}
```

---

## 5. List Datasets

**Endpoint:** `GET /api/datasets`

**Description:** List all uploaded datasets

**Example (Python):**
```python
import requests

url = "http://localhost:5000/api/datasets"
response = requests.get(url)
print(response.json())
```

**Response:**
```json
{
  "total_datasets": 3,
  "datasets": [
    {
      "dataset_id": "uuid1",
      "filename": "data1.csv",
      "processed": true,
      "total_rows": 100,
      "total_columns": 5
    }
  ]
}
```

---

## Error Responses

All endpoints may return error responses:

**400 Bad Request:**
```json
{
  "error": "No file provided"
}
```

**404 Not Found:**
```json
{
  "error": "Dataset not found"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Error message here"
}
```

---

## Complete Workflow Example
```python
import requests

BASE_URL = "http://localhost:5000"

# Step 1: Upload dataset
with open('my_data.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post(f"{BASE_URL}/api/upload", files=files)
    result = response.json()
    dataset_id = result['dataset_id']
    print(f"Uploaded: {dataset_id}")

# Step 2: Process dataset
payload = {
    "dataset_id": dataset_id,
    "n_neighbors": 5,
    "protected_attributes": ["gender"]
}
response = requests.post(f"{BASE_URL}/api/process", json=payload)
result = response.json()
print(f"Processed: {result['message']}")
print(f"Imputed values: {result['imputation']['stats']['values_imputed']}")

# Step 3: Get results
response = requests.get(f"{BASE_URL}/api/results/{dataset_id}")
result = response.json()
print(f"Output file: {result['output_file']}")
```

---

## Notes for Frontend Developers

- **CORS is enabled** for `http://localhost:3000` (React) and `http://localhost:8501` (Streamlit)
- All responses are JSON
- File uploads use `multipart/form-data`
- Other requests use `application/json`
- Maximum file size: 16MB
- Dataset ID is a UUID string