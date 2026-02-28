# Handoff Guide for Teammates

## Project Overview

This is the **ML Backend** for the Fairness Imputation System. It provides a REST API that:
- Loads and validates datasets
- Fills missing values using KNN imputation
- Audits datasets for algorithmic bias
- Returns clean, bias-checked datasets

**Your job:** Build a frontend (Streamlit/React) that calls this API.

---

## What's Already Built

✅ **Data Ingestion Module** - Validates and loads CSV/Excel files  
✅ **KNN Imputation Engine** - Fills missing values intelligently  
✅ **Fairness Auditing Module** - Detects bias in protected attributes  
✅ **Flask REST API** - Exposes all functionality via HTTP endpoints  

**You don't need to understand the ML code - just call the API!**

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/fairness-imputation-system.git
cd fairness-imputation-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the API
```bash
python run.py
```

You should see:
```
============================================================
Fairness Imputation System API
============================================================
API running at: http://localhost:5000
Health check: http://localhost:5000/health
============================================================
```

### 6. Test the API

Open a new terminal and run:
```bash
python test_stage5.py
```

If all tests pass, you're ready to go! ✅

---

## Project Structure
```
fairness-imputation-system/
├── backend/                    ← ML Backend (already complete)
│   ├── data_processing/        ← Data validation & loading
│   ├── imputation/             ← KNN imputation
│   ├── fairness_audit/         ← Bias detection
│   └── api/                    ← Flask REST API
├── data/
│   ├── raw/                    ← Uploaded files go here
│   ├── processed/              ← Intermediate files
│   └── outputs/                ← Final processed datasets
├── docs/                       ← Documentation
├── tests/                      ← Test files
└── run.py                      ← Start the API with this
```

---

## How to Use the API

### Quick Start Example
```python
import requests

BASE_URL = "http://localhost:5000"

# 1. Upload a dataset
with open('your_data.csv', 'rb') as f:
    response = requests.post(f"{BASE_URL}/api/upload", files={'file': f})
    dataset_id = response.json()['dataset_id']

# 2. Process it
payload = {"dataset_id": dataset_id}
response = requests.post(f"{BASE_URL}/api/process", json=payload)
results = response.json()

# 3. Get the clean dataset
print(f"Output file: {results['output_file']}")
print(f"Values imputed: {results['imputation']['stats']['values_imputed']}")
```

**See `API_DOCUMENTATION.md` for complete endpoint details.**

---

## What You Need to Build

### Frontend Requirements

Your frontend should:

1. **File Upload Interface**
   - Let users upload CSV/Excel files
   - Call `POST /api/upload`
   - Display upload success + basic stats

2. **Processing Interface**
   - Button to start processing
   - Call `POST /api/process` with dataset_id
   - Show loading indicator

3. **Results Display**
   - Show imputation statistics
   - Show fairness audit results
   - Display charts/visualizations
   - Provide download link for processed dataset

### Suggested Tools

- **Streamlit:** Fast, Python-based, easy UI
- **React:** More control, better for complex UIs
- **Plotly:** For interactive charts

---

## API Endpoints Summary

| Endpoint | What It Does |
|----------|--------------|
| `POST /api/upload` | Upload dataset, get dataset_id |
| `POST /api/process` | Process dataset (imputation + audit) |
| `GET /api/results/{id}` | Get results for a dataset |
| `GET /api/datasets` | List all uploaded datasets |

**Full details:** See `API_DOCUMENTATION.md`

---

## Example Workflow
```
User uploads CSV
    ↓
Frontend calls /api/upload
    ↓
Backend validates & returns dataset_id
    ↓
Frontend displays basic stats
    ↓
User clicks "Process"
    ↓
Frontend calls /api/process
    ↓
Backend runs imputation + fairness audit
    ↓
Frontend displays results + charts
    ↓
User downloads processed dataset
```

---

## Testing the API

### Manual Testing with cURL
```bash
# Upload a file
curl -X POST http://localhost:5000/api/upload \
  -F "file=@data/raw/test_data.csv"

# Process dataset (use the ID from above)
curl -X POST http://localhost:5000/api/process \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "your-uuid-here"}'
```

### Python Testing
```bash
python test_stage5.py
```

---

## Common Issues & Solutions

### Issue: API not starting
**Solution:** Make sure virtual environment is activated and dependencies installed
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: "Module not found" error
**Solution:** Make sure you're in the project root directory
```bash
cd fairness-imputation-system
python run.py
```

### Issue: CORS errors in browser
**Solution:** The API already has CORS enabled for localhost:3000 and localhost:8501. If using a different port, update `backend/api/config.py`:
```python
CORS_ORIGINS = ['http://localhost:YOUR_PORT']
```

### Issue: File upload fails
**Solution:** 
- Check file is CSV or Excel (.csv, .xlsx, .xls)
- File must be under 16MB
- File must have at least 10 rows and 2 columns

---

## Questions?

- **API Documentation:** See `docs/API_DOCUMENTATION.md`
- **Usage Examples:** See `docs/USAGE_EXAMPLES.md`
- **Issues:** Create a GitHub issue or contact Atharva

---

## Timeline

**Your tasks (Month 2):**
- Week 1: Build file upload UI
- Week 2: Build results display + visualizations
- Week 3: Create dashboard with charts
- Week 4: Testing + final presentation

**The ML backend is complete and ready to use!** 🚀