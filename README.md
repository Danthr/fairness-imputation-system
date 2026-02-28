# Automated Data Imputation and Algorithmic Fairness Auditing System

**A complete ML backend system for cleaning datasets and detecting algorithmic bias.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎓 Team Members

- **Atharva Dange** (1032221013) - ML Backend & Core Engine (Owner)
- **Ritika Palai** (1032221042)
- **Raeva Dashputre** (1032221426)
- **Khushi Bhangdia** (1032220317)

**Panel:** CSBS-A  
**Institution:** MIT WPU Pune

---

## 🚀 Project Overview

This system provides a REST API for:
- **Data Validation** - Ensures datasets meet quality standards
- **Missing Value Imputation** - Fills missing data using K-Nearest Neighbors
- **Fairness Auditing** - Detects algorithmic bias in protected attributes
- **Transparency Reports** - Generates detailed audit reports

**Use Case:** Prepare ML training data that is both statistically complete and ethically sound.

---

## ✨ Features

✅ **Automated KNN Imputation** - Intelligently fills missing values  
✅ **Baseline Comparison** - Benchmarks KNN vs mean/median imputation  
✅ **Fairness Metrics** - Disparate Impact, Demographic Parity, Statistical Parity  
✅ **Protected Attribute Detection** - Auto-detects sensitive columns (gender, race, age)  
✅ **REST API** - Easy integration with any frontend  
✅ **Multi-format Support** - CSV, Excel (XLSX, XLS)  

---

## 🏗️ Architecture
```
┌─────────────────┐
│   Frontend      │  ← Your teammates build this
│  (Streamlit/    │
│    React)       │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   Flask API     │  ✅ COMPLETE
│   (Port 5000)   │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬────────────┐
    ▼         ▼          ▼            ▼
┌────────┐ ┌─────┐ ┌──────────┐ ┌─────────┐
│ Data   │ │ KNN │ │ Fairness │ │ Report  │  ✅ ALL COMPLETE
│ Ingest │ │Imput│ │  Audit   │ │   Gen   │
└────────┘ └─────┘ └──────────┘ └─────────┘
```

---

## 📦 Tech Stack

- **Language:** Python 3.9+
- **ML Framework:** Scikit-Learn
- **Fairness Library:** Fairlearn
- **Web Framework:** Flask + Flask-CORS
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib
- **Testing:** Pytest

---

## 🔧 Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git

### Installation Steps

1. **Clone the repository**
```bash
   git clone https://github.com/YOUR-USERNAME/fairness-imputation-system.git
   cd fairness-imputation-system
```

2. **Create virtual environment**
```bash
   python -m venv venv
```

3. **Activate virtual environment**
   
   **Windows:**
```bash
   venv\Scripts\activate
```
   
   **Mac/Linux:**
```bash
   source venv/bin/activate
```

4. **Install dependencies**
```bash
   pip install -r requirements.txt
```

5. **Run the API**
```bash
   python run.py
```

6. **Verify installation**
```bash
   # In a new terminal
   python test_stage5.py
```

✅ If all tests pass, you're ready to go!

---

## 🎯 Quick Start

### Using the API (Python)
```python
import requests

BASE_URL = "http://localhost:5000"

# 1. Upload dataset
with open('my_data.csv', 'rb') as f:
    response = requests.post(f"{BASE_URL}/api/upload", files={'file': f})
    dataset_id = response.json()['dataset_id']

# 2. Process dataset
payload = {"dataset_id": dataset_id}
response = requests.post(f"{BASE_URL}/api/process", json=payload)
results = response.json()

# 3. Get results
print(f"Values imputed: {results['imputation']['stats']['values_imputed']}")
print(f"Output file: {results['output_file']}")
```

**See full documentation:** [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md)

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [API Documentation](docs/API_DOCUMENTATION.md) | Complete API endpoint reference |
| [Handoff Guide](docs/HANDOFF_GUIDE.md) | Setup instructions for teammates |
| [Usage Examples](docs/USAGE_EXAMPLES.md) | Code examples in Python, React, Streamlit |

---

## 📁 Project Structure
```
fairness-imputation-system/
├── backend/                    # ML Backend (COMPLETE)
│   ├── data_processing/        # Data validation & loading
│   │   ├── validator.py
│   │   └── ingestion.py
│   ├── imputation/             # KNN imputation engine
│   │   └── knn_imputer.py
│   ├── fairness_audit/         # Bias detection
│   │   ├── metrics.py
│   │   └── auditor.py
│   └── api/                    # Flask REST API
│       ├── app.py
│       ├── routes.py
│       └── config.py
├── data/
│   ├── raw/                    # Uploaded datasets
│   ├── processed/              # Intermediate files
│   └── outputs/                # Final processed datasets
├── docs/                       # Documentation
│   ├── API_DOCUMENTATION.md
│   ├── HANDOFF_GUIDE.md
│   └── USAGE_EXAMPLES.md
├── tests/                      # Test files
├── requirements.txt            # Python dependencies
├── run.py                      # Start API server
└── README.md                   # This file
```

---

## 🧪 Testing

### Run All Tests
```bash
# Test data ingestion
python test_stage2.py

# Test KNN imputation
python test_stage3.py

# Test fairness auditing
python test_stage4.py

# Test API endpoints
python test_stage5.py
```

### Manual API Testing
```bash
# Health check
curl http://localhost:5000/health

# Upload file
curl -X POST http://localhost:5000/api/upload \
  -F "file=@data/raw/test_data.csv"
```

---

## 🎨 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check API health |
| POST | `/api/upload` | Upload dataset |
| POST | `/api/process` | Process dataset (imputation + audit) |
| GET | `/api/results/{id}` | Get processing results |
| GET | `/api/datasets` | List all datasets |

**Full API docs:** See [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md)

---

## 📊 Development Status

### ✅ Completed (Month 1)
- [x] Stage 1: Project Setup
- [x] Stage 2: Data Ingestion Module
- [x] Stage 3: KNN Imputation Engine
- [x] Stage 4: Fairness Auditing Module
- [x] Stage 5: Flask REST API
- [x] Stage 6: Documentation & Handoff

### ⏳ In Progress (Month 2)
- [ ] Frontend UI (Streamlit/React)
- [ ] Visualization Dashboard
- [ ] Report Generation
- [ ] Integration Testing
- [ ] Final Presentation

---

## 🤝 For Teammates

**The ML backend is complete and ready to use!**

1. Follow the [Handoff Guide](docs/HANDOFF_GUIDE.md) to set up the project
2. Read the [API Documentation](docs/API_DOCUMENTATION.md) to understand endpoints
3. Check [Usage Examples](docs/USAGE_EXAMPLES.md) for code samples
4. Build your frontend to call the API

**Need help?** Contact Atharva Dange

---

## 📈 Performance Metrics

**Imputation Performance:**
- Average processing time: 0.01-0.05s per dataset
- Supports datasets up to 100K rows
- Handles both numerical and categorical data
- 100% missing value coverage with KNN

**Fairness Metrics Implemented:**
- Disparate Impact (4/5ths rule)
- Demographic Parity
- Statistical Parity Difference

---

## 🔮 Future Enhancements

- Database integration (currently uses in-memory storage)
- Advanced imputation methods (MICE, Deep Learning)
- Additional fairness metrics (Equal Opportunity)
- Batch processing support
- Real-time progress updates via WebSocket
- User authentication & authorization

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** - Test datasets
- **Scikit-Learn** - ML algorithms
- **Fairlearn** - Fairness metrics
- **MIT WPU** - Project guidance

---

## 📧 Contact

**Atharva Dange**  
📧 1032221013@mitwpu.edu.in  
🔗 [LinkedIn](https://linkedin.com/in/atharva-dange/)  
💻 [GitHub](https://github.com/Danthr)

---

**⭐ Star this repo if you find it helpful!**