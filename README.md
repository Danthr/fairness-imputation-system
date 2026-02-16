# Automated Data Imputation and Algorithmic Fairness Auditing System

## Team Members
- **Atharva Dange** (1032221013) - ML Backend & Core Engine
- **Ritika Palai** (1032221042)
- **Raeva Dashputre** (1032221426)
- **Khushi Bhangdia** (1032220317)

## Project Overview
A centralized automated framework that addresses both data quality and ethical integrity in ML pipelines by performing KNN-based imputation for missing values and auditing datasets for algorithmic bias.

## Tech Stack
- **Language:** Python 3.9+
- **ML Framework:** Scikit-Learn
- **Fairness:** Fairlearn
- **API:** Flask
- **Visualization:** Plotly, Matplotlib

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/fairness-imputation-system.git
cd fairness-imputation-system
```

### 2. Create virtual environment
```bash
python -m venv venv
```

### 3. Activate virtual environment
**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

## Project Structure
```
fairness-imputation-system/
├── backend/
│   ├── data_processing/    # Data ingestion and validation
│   ├── imputation/         # KNN imputation engine
│   ├── fairness_audit/     # Fairness metrics
│   ├── api/                # Flask REST API
│   └── utils/              # Helper functions
├── data/
│   ├── raw/                # Original datasets
│   ├── processed/          # Preprocessed data
│   └── outputs/            # Final AI-ready datasets
└── tests/                  # Unit tests
```

## Development Progress
- [x] Project setup
- [ ] Data ingestion module
- [ ] KNN imputation engine
- [ ] Fairness auditing module
- [ ] Flask API
- [ ] Report generation

