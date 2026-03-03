# Frontend Handoff Guide
## Fairness Imputation System — Stage 7

**Date:** March 2026  
**Backend Developer:** Atharva Dange  
**Status:** Backend complete, ready for frontend integration

---

## What This System Does

A machine learning backend that:
1. Accepts datasets with missing values (CSV/Excel)
2. Assesses data quality before imputation
3. Imputes missing values using KNN, MICE, or Random Forest
4. Evaluates imputation accuracy (RMSE, MAE)
5. Audits the imputed dataset for fairness across protected attributes (gender, age, etc.)

---

## Getting the Backend Running
```bash
# 1. Navigate to project folder
cd fairness-imputation-system

# 2. Activate virtual environment
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies (first time only)
pip install -r requirements.txt

# 4. Start the API
python run.py

# API is now running at http://localhost:5000
```

---

## Quick Sanity Check
```bash
curl http://localhost:5000/health
# Expected: {"status": "healthy", "message": "API is running"}
```

---

## CORS

CORS is already enabled for:
- `http://localhost:3000` (React)
- `http://localhost:8501` (Streamlit)

No CORS configuration needed on the frontend side.

---

## The 5 Endpoints You Need

### Overview

| # | Method | Endpoint | When to call |
|---|--------|----------|--------------|
| 1 | POST | `/api/upload` | User selects a file |
| 2 | GET | `/api/quality/{dataset_id}` | After upload, before processing |
| 3 | POST | `/api/process` | User clicks "Run Imputation" |
| 4 | GET | `/api/results/{dataset_id}` | Fetch stored results anytime |
| 5 | GET | `/api/datasets` | Dashboard listing all uploads |

---

### 1. Upload — `POST /api/upload`

**When:** User selects a CSV/Excel file.

**Request:**
```javascript
const formData = new FormData();
formData.append('file', selectedFile);

const response = await fetch('http://localhost:5000/api/upload', {
    method: 'POST',
    body: formData
});
const data = await response.json();
const datasetId = data.dataset_id;  // save this — needed for all other calls
```

**Key response fields:**
```json
{
  "dataset_id": "uuid-string",
  "filename": "data.csv",
  "message": "File uploaded successfully",
  "stats": {
    "total_rows": 100,
    "total_columns": 5,
    "missing_cells": 10,
    "missing_percentage": 2.0,
    "numerical_columns": ["age", "income"],
    "categorical_columns": ["gender", "education"]
  }
}
```

**What to show in UI:**
- Success message with filename
- Summary card: rows, columns, missing values count
- "Check Data Quality" button → triggers endpoint 2

---

### 2. Data Quality — `GET /api/quality/{dataset_id}`

**When:** After upload, before the user runs imputation.
This tells them how healthy their data is before processing.

**Request:**
```javascript
const response = await fetch(`http://localhost:5000/api/quality/${datasetId}`);
const data = await response.json();
```

**Key response fields:**
```json
{
  "data_quality": {
    "overall_score": 0.87,
    "overall_grade": "B",
    "recommendation": "Issues found: high missing rate (35%) — imputation strongly recommended.",
    "dimension_scores": {
      "completeness": 0.65,
      "validity":     1.0,
      "consistency":  1.0,
      "uniqueness":   1.0
    }
  }
}
```

**What to show in UI:**
- Overall score (large number) + grade (A/B/C/D/F)
- 4 dimension score cards or a radar/spider chart
- Recommendation text (already human-readable, just display it)
- "Run Imputation" button → triggers endpoint 3

**Grade colour suggestions:**
| Grade | Colour |
|-------|--------|
| A | Green |
| B | Light green |
| C | Yellow |
| D | Orange |
| F | Red |

---

### 3. Process — `POST /api/process`

**When:** User clicks "Run Imputation".

**Request:**
```javascript
const response = await fetch('http://localhost:5000/api/process', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        dataset_id: datasetId,
        method: selectedMethod,          // "knn" | "mice" | "random_forest" | "compare_all"
        n_neighbors: 5,
        include_confidence: true,
        include_quality: true,
        include_data_quality: true,
        protected_attributes: ["gender", "age"]
    })
});
const data = await response.json();
```

**Available imputation methods:**

| Value | Display Name | Notes |
|-------|-------------|-------|
| `"knn"` | KNN (default) | Fast, good general purpose |
| `"mice"` | MICE | Best accuracy, slower |
| `"random_forest"` | Random Forest | Most powerful, slowest |
| `"compare_all"` | Compare All | Runs all methods, returns comparison |

**Key response fields:**
```json
{
  "message": "Processing complete",
  "imputation": {
    "stats": {
      "method": "KNN",
      "values_imputed": 8,
      "missing_before": 8,
      "missing_after": 0,
      "execution_time_seconds": 0.01
    }
  },
  "confidence_scores": {
    "age": [
      { "row_index": 2, "imputed_value": 34.5, "confidence": 0.85 }
    ]
  },
  "quality_metrics": {
    "best_method": "mice",
    "ranking": ["mice", "knn", "rf", "mean"],
    "summary": {
      "knn":  { "rmse": 3.21, "mae": 2.45 },
      "mice": { "rmse": 2.87, "mae": 2.01 }
    }
  },
  "data_quality": {
    "overall_score": 0.87,
    "overall_grade": "B"
  },
  "fairness_audit": {
    "gender": {
      "overall_assessment": {
        "is_fair": true,
        "summary": "Dataset passes fairness checks"
      }
    }
  },
  "output_file": "data/outputs/uuid_processed.csv"
}
```

**What to show in UI:**
- Imputation summary (values filled, method used, time taken)
- Confidence scores table (row, column, imputed value, confidence %)
- Quality metrics: method ranking bar chart (RMSE comparison)
- Fairness audit: pass/fail per protected attribute
- Download button for processed CSV

---

### 4. Get Results — `GET /api/results/{dataset_id}`

**When:** User revisits a previously processed dataset, or you want to
reload results without reprocessing.

**Request:**
```javascript
const response = await fetch(`http://localhost:5000/api/results/${datasetId}`);
const data = await response.json();
```

**Note:** Returns `status: "uploaded"` if dataset hasn't been processed yet.

---

### 5. List Datasets — `GET /api/datasets`

**When:** Dashboard page showing all uploaded datasets.

**Request:**
```javascript
const response = await fetch('http://localhost:5000/api/datasets');
const data = await response.json();
// data.datasets = array of all uploads
```

---

## Recommended UI Flow
```
Upload File
    ↓
Show file stats (rows, columns, missing values)
    ↓
Auto-call /api/quality → Show quality scorecard
    ↓
User selects imputation method + options
    ↓
Call /api/process → Show results dashboard
    ↓
Results: imputation summary | quality metrics | fairness audit
    ↓
Download processed CSV
```

---

## Suggested Pages / Components

| Page | Endpoints used |
|------|---------------|
| Upload page | `/api/upload` |
| Quality dashboard | `/api/quality/{id}` |
| Processing page | `/api/process` |
| Results dashboard | `/api/results/{id}` |
| History / all datasets | `/api/datasets` |

---

## Error Handling

All endpoints return errors in this format:
```json
{ "error": "Error message here" }
```

HTTP status codes:
- `400` — bad request (missing file, missing dataset_id)
- `404` — dataset not found
- `500` — server error
```javascript
const data = await response.json();
if (!response.ok || data.error) {
    showErrorMessage(data.error);
    return;
}
```

---

## Optional Fields in /api/process Response

These fields are **only present** when you request them:

| Response field | Request flag needed |
|----------------|-------------------|
| `confidence_scores` | `"include_confidence": true` |
| `method_comparison` | `"method": "compare_all"` |
| `quality_metrics` | `"include_quality": true` |
| `data_quality` | `"include_data_quality": true` |

Always check if these fields exist before accessing them:
```javascript
if (data.quality_metrics) {
    showQualityMetrics(data.quality_metrics);
}
if (data.confidence_scores) {
    showConfidenceTable(data.confidence_scores);
}
```

---
