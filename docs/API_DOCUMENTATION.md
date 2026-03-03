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
| GET | `/api/quality/{dataset_id}` | Data quality scoring (NEW - Stage 7) |
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

**Description:** Run imputation and fairness audit on an uploaded dataset.
Supports multiple imputation methods, confidence scores, imputation quality
metrics, and data quality scoring — all via optional flags.

**Request Body:**
```json
{
  "dataset_id": "uuid-from-upload",
  "method": "knn",
  "n_neighbors": 5,
  "include_confidence": false,
  "include_quality": false,
  "quality_methods": ["knn", "mice", "rf", "mean"],
  "include_cv": false,
  "include_data_quality": false,
  "protected_attributes": ["gender", "race"]
}
```

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_id` | string | required | UUID from `/api/upload` |
| `method` | string | `"knn"` | Imputation method: `knn`, `mice`, `random_forest`, `compare_all` |
| `n_neighbors` | int | `5` | Number of neighbors for KNN |
| `include_confidence` | bool | `false` | Include per-value confidence scores in response |
| `include_quality` | bool | `false` | Run imputation quality metrics (RMSE, MAE, CV) |
| `quality_methods` | list | `["knn","mice","rf","mean"]` | Methods to compare when `include_quality=true` |
| `include_cv` | bool | `false` | Include cross-validation scores when `include_quality=true` |
| `include_data_quality` | bool | `false` | Run data quality scoring (completeness, validity, consistency, uniqueness) |
| `protected_attributes` | list | auto-detect | Columns to audit for fairness |

**Example - Basic KNN (Python):**
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

**Example - MICE with confidence scores (Python):**
```python
payload = {
    "dataset_id": "your-dataset-id",
    "method": "mice"
}
response = requests.post(url, json=payload)
```

**Example - Full pipeline with all features (Python):**
```python
payload = {
    "dataset_id": "your-dataset-id",
    "method": "knn",
    "include_confidence": True,
    "include_quality": True,
    "include_cv": True,
    "include_data_quality": True,
    "protected_attributes": ["gender"]
}
response = requests.post(url, json=payload)
```

**Example - Compare all methods (cURL):**
```bash
curl -X POST http://localhost:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{"dataset_id":"uuid-here","method":"compare_all"}'
```

**Response (base):**
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

**Optional response field - `confidence_scores`** (when `include_confidence=true`):
```json
{
  "confidence_scores": {
    "age": [
      {
        "row_index": 2,
        "imputed_value": 34.5,
        "confidence": 0.85
      }
    ]
  }
}
```

**Optional response field - `method_comparison`** (when `method=compare_all`):
```json
{
  "method_comparison": {
    "knn":           { "values_imputed": 10, "execution_time_seconds": 0.01 },
    "mice":          { "values_imputed": 10, "execution_time_seconds": 0.08 },
    "random_forest": { "values_imputed": 10, "execution_time_seconds": 0.24 },
    "mean":          { "values_imputed": 10, "execution_time_seconds": 0.01 },
    "recommended_method": "knn"
  }
}
```

**Optional response field - `quality_metrics`** (when `include_quality=true`):
```json
{
  "quality_metrics": {
    "ranking": ["mice", "knn", "rf", "mean"],
    "best_method": "mice",
    "summary": {
      "knn":  { "rmse": 3.21, "mae": 2.45 },
      "mice": { "rmse": 2.87, "mae": 2.01 },
      "rf":   { "rmse": 3.54, "mae": 2.78 },
      "mean": { "rmse": 4.12, "mae": 3.33 }
    },
    "methods": {
      "knn": {
        "method": "knn",
        "rmse": 3.21,
        "mae": 2.45,
        "n_evaluated": 8,
        "eval_time_s": 0.005,
        "per_column": {
          "age":    { "rmse": 2.10, "mae": 1.80, "n_evaluated": 4 },
          "income": { "rmse": 4.32, "mae": 3.10, "n_evaluated": 4 }
        },
        "cross_validation": {
          "cv_rmse_mean": 3.45,
          "cv_rmse_std":  0.32,
          "cv_mae_mean":  2.67,
          "cv_mae_std":   0.28,
          "fold_scores": [
            { "fold": 1, "rmse": 3.12, "mae": 2.45 },
            { "fold": 2, "rmse": 3.78, "mae": 2.89 }
          ]
        }
      }
    }
  }
}
```

**Optional response field - `data_quality`** (when `include_data_quality=true`):
```json
{
  "data_quality": {
    "overall_score": 0.87,
    "overall_grade": "B",
    "recommendation": "Issues found: high missing rate (35.0%) — imputation strongly recommended.",
    "dimension_scores": {
      "completeness": 0.65,
      "validity":     1.0,
      "consistency":  1.0,
      "uniqueness":   1.0
    },
    "completeness": {
      "score": 0.65,
      "missing_cells": 7,
      "missing_pct": 35.0,
      "grade": "D",
      "per_column": {
        "age":    { "missing": 2, "present": 8, "score": 0.80 },
        "income": { "missing": 3, "present": 7, "score": 0.70 }
      }
    },
    "validity": {
      "score": 1.0,
      "total_invalid": 0,
      "grade": "A",
      "per_column": {}
    },
    "consistency": {
      "score": 1.0,
      "n_violations": 0,
      "grade": "A",
      "violations": []
    },
    "uniqueness": {
      "score": 1.0,
      "duplicate_rows": 0,
      "duplicate_pct": 0.0,
      "grade": "A"
    }
  }
}
```

---

## 4. Data Quality Scoring (NEW - Stage 7)

**Endpoint:** `GET /api/quality/{dataset_id}`

**Description:** Run a standalone data quality assessment on an uploaded dataset.
Use this **before** `/api/process` to check data health and decide which
imputation method is most appropriate.

**Example (Python):**
```python
import requests

url = "http://localhost:5000/api/quality/your-dataset-id"
response = requests.get(url)
print(response.json())
```

**Example (cURL):**
```bash
curl http://localhost:5000/api/quality/your-dataset-id
```

**Response:**
```json
{
  "dataset_id": "uuid",
  "filename": "data.csv",
  "message": "Data quality assessment complete",
  "data_quality": {
    "overall_score": 0.87,
    "overall_grade": "B",
    "recommendation": "Issues found: high missing rate (35.0%) — imputation strongly recommended.",
    "dimension_scores": {
      "completeness": 0.65,
      "validity":     1.0,
      "consistency":  1.0,
      "uniqueness":   1.0
    },
    "completeness":  { "score": 0.65, "grade": "D", "missing_cells": 7, "missing_pct": 35.0 },
    "validity":      { "score": 1.0,  "grade": "A", "total_invalid": 0 },
    "consistency":   { "score": 1.0,  "grade": "A", "n_violations": 0 },
    "uniqueness":    { "score": 1.0,  "grade": "A", "duplicate_rows": 0 },
    "weights_used":  { "completeness": 0.35, "validity": 0.25, "consistency": 0.25, "uniqueness": 0.15 }
  }
}
```

**Grade Scale:**

| Score | Grade | Meaning |
|-------|-------|---------|
| 0.95 – 1.00 | A | Excellent |
| 0.85 – 0.94 | B | Good |
| 0.70 – 0.84 | C | Acceptable |
| 0.50 – 0.69 | D | Poor |
| 0.00 – 0.49 | F | Critical |

---

## 5. Get Results

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

## 6. List Datasets

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
      "total_columns": 5,
      "method_used": "knn"
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

# Step 2: Check data quality before processing
response = requests.get(f"{BASE_URL}/api/quality/{dataset_id}")
quality = response.json()
print(f"Data quality score: {quality['data_quality']['overall_score']}")
print(f"Recommendation: {quality['data_quality']['recommendation']}")

# Step 3: Process dataset with full features
payload = {
    "dataset_id": dataset_id,
    "method": "knn",
    "n_neighbors": 5,
    "include_confidence": True,
    "include_quality": True,
    "include_data_quality": True,
    "protected_attributes": ["gender"]
}
response = requests.post(f"{BASE_URL}/api/process", json=payload)
result = response.json()
print(f"Processed: {result['message']}")
print(f"Imputed values: {result['imputation']['stats']['values_imputed']}")
print(f"Best imputation method: {result['quality_metrics']['best_method']}")

# Step 4: Get stored results anytime
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
- All optional response fields (`confidence_scores`, `method_comparison`, `quality_metrics`, `data_quality`) are only present in the response when their corresponding request flag is set to `true`
- Call `/api/quality/{dataset_id}` before `/api/process` to assess data health first