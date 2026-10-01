# MLOps-Based Banking Fraud Detection and Real-Time Transaction Risk Prediction System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-228B22?style=flat&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-Configured-2088FF?style=flat&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![Tests](https://img.shields.io/badge/Tests-40%2F40%20Passed-brightgreen?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org)

An enterprise-grade, end-to-end Machine Learning Operations (MLOps) platform designed for real-time banking transaction risk assessment and fraud detection. The system pairs an optimized 79-feature XGBoost classifier with production-grade FastAPI services, automated MLflow experiment tracking, thread-safe prediction logging, real-time statistical data drift detection using Population Stability Index (PSI), multi-stage Docker containerization, GitHub Actions CI/CD workflows, and a reactive dark-themed cybersecurity operations dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Key Features](#4-key-features)
5. [System Architecture](#5-system-architecture)
6. [Dataset](#6-dataset)
7. [Dataset Validation](#7-dataset-validation)
8. [Preprocessing](#8-preprocessing)
9. [Feature Engineering](#9-feature-engineering)
10. [Model Training](#10-model-training)
11. [Model Evaluation](#11-model-evaluation)
12. [Production Inference](#12-production-inference)
13. [FastAPI Backend](#13-fastapi-backend)
14. [React Frontend](#14-react-frontend)
15. [MLOps Architecture](#15-mlops-architecture)
16. [MLflow Experiment Tracking](#16-mlflow-experiment-tracking)
17. [Prediction Logging](#17-prediction-logging)
18. [Runtime Monitoring](#18-runtime-monitoring)
19. [PSI Data Drift Detection](#19-psi-data-drift-detection)
20. [Docker Containerization](#20-docker-containerization)
21. [CI/CD Pipelines](#21-cicd-pipelines)
22. [Project Structure](#22-project-structure)
23. [Installation](#23-installation)
24. [Local Development](#24-local-development)
25. [Running MLflow](#25-running-mlflow)
26. [Running Backend](#26-running-backend)
27. [Running Frontend](#27-running-frontend)
28. [Running with Docker](#28-running-with-docker)
29. [API Documentation](#29-api-documentation)
30. [Testing & Quality Assurance](#30-testing--quality-assurance)
31. [Model Performance](#31-model-performance)
32. [Limitations](#32-limitations)
33. [Future Improvements](#33-future-improvements)
34. [Deployment Status](#34-deployment-status)

---

## 1. Project Overview

The **MLOps-Based Banking Fraud Detection and Real-Time Transaction Risk Prediction System** addresses financial fraud by integrating machine learning models with production-grade software engineering and MLOps practices. The platform processes high-velocity transaction streams, scores fraud probabilities under 30 milliseconds, stratifies operational risk into actionable tiers, logs all evaluations safely, and continuously monitors input data distributions for statistical drift.

## 2. Problem Statement

Financial institutions process millions of transactions daily where fraudulent activity accounts for less than 1.5% of total volume. Key challenges include:
- **Severe Class Imbalance:** Fraudulent transactions represent approximately 1.2% of transactions, leading naive classifiers to achieve high accuracy while failing to capture actual fraud.
- **Latency Constraints:** Fraud assessment must execute synchronously without adding measurable latency to card payments or bank transfers (< 50 ms).
- **Data Distribution Drift:** Fraud patterns and consumer spending habits evolve rapidly across merchant categories, amounts, and operating hours.
- **Production Reproducibility:** Bridging the gap between offline Jupyter notebooks and deterministic, testable, containerized production systems.

## 3. Objectives

- Develop a high-recall XGBoost model capturing > 95% of fraudulent events while maintaining acceptable precision.
- Standardize a 79-feature deterministic production inference pipeline with zero training-serving skew.
- Expose hardened, asynchronous REST API endpoints with comprehensive Pydantic domain validation.
- Implement automated MLOps experiment tracking and local model governance with MLflow.
- Deploy resilient, thread-safe prediction logging and dynamic runtime metrics computation.
- Implement statistical Population Stability Index (PSI) data drift detection against reference training baselines.
- Provide a responsive, cybersecurity-themed React 19 dashboard for real-time inference and observability.
- Package the platform into production Docker containers and automate testing via GitHub Actions CI/CD.

## 4. Key Features

- **XGBoost Classifier:** Optimized with `scale_pos_weight=45.0` to maximize fraud sensitivity on imbalanced streams.
- **Deterministic 79-Feature Space:** Strict categorical dummy alignment, log transforms, and z-score standardizations.
- **Calibrated Risk Tiers:** LOW (< 0.20 auto-approval), MEDIUM (0.20–0.50 step-up authentication), and HIGH (≥ 0.50 auto-block).
- **Thread-Safe Prediction Logger:** Writes 12-attribute JSONL records asynchronously with non-blocking error absorption.
- **Population Stability Index (PSI):** Quantifies distribution shifts across 6 raw input features with industry standard thresholds.
- **Full-Stack Containerization:** Multi-stage Docker build for frontend (Nginx) and slim Python image for backend.
- **Comprehensive Test Suite:** 40/40 passing automated unit and integration tests across inference, API, and monitoring.

---

## 5. System Architecture

### Text Architecture Diagram

```
User / Client
     │
     ▼
React Frontend (Port 5173 / Port 80)
     │
   HTTP
     ▼
FastAPI Gateway (Port 8000)
     │
     ├───────────────────────────────────┐
     │                                   │
     ▼                                   ▼
Pydantic Validation                 Monitoring Service (GET /monitoring)
     │
     ▼
Prediction Service (app/services/)
     │
     ▼
Preprocessing Pipeline (src/preprocessing.py)
     │
     ▼
Feature Engineering (src/feature_engineering.py)
     │
     ▼
XGBoost Booster (models/xgboost_model.pkl)
     │
     ▼
Classification: Fraud / Legitimate (with Probability & Risk Tier)
     │
     ├───────────────────────────────────┐
     │                                   │
     ▼                                   ▼
Prediction Logger (Thread-Safe)     Monitoring Telemetry
     │                                   │
     ▼                                   ▼
logs/predictions.jsonl              PSI Data Drift Engine (GET /monitoring/drift)
                                         │
                                         ▼
                                    Baseline Data (594,643 Training Samples)

MLflow Tracking Server (Port 5000)
     │
     ├── Parameters (max_depth=6, learning_rate=0.1, n_estimators=100)
     ├── Evaluation Metrics (ROC-AUC=0.9975, PR-AUC=0.8951, Recall=0.9701)
     ├── Model Metadata (models/model_metadata.json)
     └── Serialized Artifacts (xgboost_model.pkl, preprocessor.pkl)
```

---

## 6. Dataset

- **Source:** Synthetic banking payments transaction log based on actual bank transaction characteristics.
- **Total Records:** 594,643 transactions.
- **Raw Features:**
  - `step`: Simulation unit representing an hour of time (0 to 179).
  - `customer`: Customer account identifier string (e.g., `C1093113857`).
  - `age`: Customer age category (0: ≤16, 1: 17–25, 2: 26–35, 3: 36–45, 4: 46–55, 5: 56–65, 6: >65, U: Unknown).
  - `gender`: Gender identifier (`M`: Male, `F`: Female, `E`: Enterprise, `U`: Unknown).
  - `zipcodeOri` / `zipMerchant`: Constant zero-variance postal codes (dropped during validation).
  - `merchant`: Merchant unique identifier string (e.g., `M1823072687`).
  - `category`: Transaction spending category (15 merchant categories).
  - `amount`: Transaction financial amount.
  - `fraud`: Binary classification label (0 = Legitimate, 1 = Fraudulent).
- **Target Distribution:** 587,443 legitimate transactions (98.79%) vs. 7,200 fraudulent transactions (1.21%).

## 7. Dataset Validation

During Phase 3, rigorous schema and data hygiene checks verified:
- Zero null, NaN, or infinite values across all 594,643 rows.
- Dropped zero-variance constant columns (`zipcodeOri`, `zipMerchant`).
- Dropped unique identifiers (`customer`) that cause high-cardinality overfitting.
- Validated amounts are strictly non-negative ($0.0 \le \text{amount} < \infty$).
- Validated categorical sets: 15 unique categories, 50 unique merchants, 8 age brackets, and 4 gender codes.

## 8. Preprocessing

Standardized in [`src/preprocessing.py`](file:///c:/Users/Bharath/Downloads/Bank-Fraud-MLOps/src/preprocessing.py):
- Enforces deterministic one-hot dummy generation matching the training reference baselines.
- Uses `drop_first=True` logic anchored to fixed reference categories:
  - `gender`: reference category `E`
  - `category`: reference category `es_barsandrestaurants`
  - `merchant`: reference category `M1053599405`
- Preserves all known categories even when a single transaction contains only one category, guaranteeing exactly 75 output columns prior to feature engineering.

## 9. Feature Engineering

Standardized in [`src/feature_engineering.py`](file:///c:/Users/Bharath/Downloads/Bank-Fraud-MLOps/src/feature_engineering.py):
Applies 4 domain-engineered transformations using precomputed, stored training statistics:
1. `amount_log`: $\ln(1 + \text{amount})$ to compress heavy-tailed spending distributions.
2. `amount_bucket`: Categorical spending bracket (0: $\le \$10$, 1: $\$10–\$50$, 2: $\$50–\$100$, 3: $\$100–\$500$, 4: $>\$500$).
3. `amount_zscore`: Standardized amount deviation using fixed training mean ($\mu = 37.89$) and standard deviation ($\sigma = 111.40$).
4. `high_amount_flag`: Binary indicator (1 if transaction exceeds the 95th training percentile threshold of $\$97.27$, else 0).
- Final engineered feature matrix produces **exactly 79 features** in an identical schema order.

## 10. Model Training

In Phase 6, multiple classification algorithms were trained and benchmarked using stratified 80/20 train-test splits:
- **Baseline Logistic Regression:** Fast, linear baseline.
- **Random Forest Classifier:** 100 decision trees with balanced subsampling.
- **XGBoost Classifier (Selected Production Model):** Gradient boosted decision trees configured with:
  - `n_estimators`: 100
  - `max_depth`: 6
  - `learning_rate`: 0.1
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `scale_pos_weight`: 45.0 (optimizes loss gradient for the 1.2% minority class)
  - `eval_metric`: `logloss`

## 11. Model Evaluation

Evaluation across 118,929 test set transactions demonstrated superior fraud capture for XGBoost:

| Metric | Logistic Regression | Random Forest | XGBoost (Production) |
|---|---|---|---|
| **ROC-AUC** | 0.8920 | 0.9856 | **0.9975** |
| **PR-AUC** | 0.6214 | 0.8410 | **0.8951** |
| **Recall (Fraud)** | 62.45% | 88.38% | **97.01%** (1,128 / 1,162 caught) |
| **Precision (Fraud)** | 45.12% | 82.50% | **33.70%** |
| **F1-Score** | 0.5238 | 0.8534 | **0.5003** |
| **Overall Accuracy** | 98.85% | 99.65% | **97.65%** |

> **IMPORTANT:** The metrics above represent **evaluation-set benchmark metrics** on the fixed test partition. They do not represent live production performance, as incoming production transactions lack verified ground-truth labels.

---

## 12. Production Inference

Exposed in [`src/inference.py`](file:///c:/Users/Bharath/Downloads/Bank-Fraud-MLOps/src/inference.py):
- `load_model()`: Caches the `XGBClassifier` booster in memory, verifying 79 input features.
- `predict_transaction()`: Accepts a raw transaction dictionary, validates fields, transforms through preprocessing and feature engineering, and returns probability, binary classification, and operational risk tier.
- `predict_batch()`: High-throughput batch scoring.
- **Calibrated Risk Tiers:**
  - **LOW (< 0.20):** Normal transaction pattern. Negligible fraud probability. Auto-approved.
  - **MEDIUM (0.20–0.50):** Borderline suspicious behavior. Step-up authentication required (SMS/OTP/2FA).
  - **HIGH (≥ 0.50):** Classified as Fraud. Strong anomalous indicators. Auto-blocked or assigned to fraud operations review.

## 13. FastAPI Backend

Located in [`app/main.py`](file:///c:/Users/Bharath/Downloads/Bank-Fraud-MLOps/app/main.py):
- High-performance asynchronous REST API powered by Starlette and Pydantic v2.
- Lifespan context manager loads model artifacts once during startup.
- Configurable CORS middleware via `ALLOWED_ORIGINS`.
- Centralized exception handlers return clean RFC-7807 JSON errors without leaking internals.
- Interactive documentation auto-generated at `/docs` (Swagger UI) and `/redoc`.

## 14. React Frontend

Located in [`frontend/`](file:///c:/Users/Bharath/Downloads/Bank-Fraud-MLOps/frontend):
- Built with React 19, Vite 8.3, and Lucide React icons.
- Cybersecurity dark-mode interface with live system status badges and animated pulse indicators.
- **Interactive Form:** Pre-fills realistic legitimate and fraud test scenarios, validates inputs dynamically.
- **Risk Scorecard:** Renders dynamic radial gauges, confidence intervals, and decision cards.
- **MLOps Monitoring Module:** Displays real-time operational statistics and Population Stability Index (PSI) feature drift.
- **Session History:** Preserves local audit trails in browser `localStorage`.

---

## 15. MLOps Architecture

The MLOps layer guarantees model reproducibility, runtime observability, and automated failure detection:
1. **Model Governance:** Parameters, metrics, and serialized artifacts are tracked in MLflow.
2. **Telemetry Ingestion:** Non-blocking prediction logging captures incoming transaction features and latency.
3. **Continuous Monitoring:** Operational counters compute live fraud rates and risk distributions dynamically.
4. **Data Quality Surveillance:** Population Stability Index (PSI) checks for input feature drift before predictions degrade.

## 16. MLflow Experiment Tracking

- **Experiment Name:** `bank-fraud-detection`
- **Backend Store:** Local SQLite database (`sqlite:///mlflow.db`).
- **Artifact Store:** Local directory (`mlruns/`).
- **Logged Parameters:** Tree depth (`max_depth=6`), estimators (`100`), learning rate (`0.1`), subsample (`0.8`), scale pos weight (`45.0`), and all 79 feature names.
- **Logged Metrics:** ROC-AUC (`0.9975`), PR-AUC (`0.8951`), Recall (`0.9701`), F1 (`0.5003`), Precision (`0.3370`), Accuracy (`0.9765`).
- **Logged Artifacts:** `xgboost_model.pkl`, `preprocessor.pkl`, `feature_config.json`, `baseline_distribution.json`, `model_metadata.json`.

## 17. Prediction Logging

Located in [`src/monitoring/prediction_logger.py`](file:///c:/Users/Bharath/Downloads/Bank-Fraud-MLOps/src/monitoring/prediction_logger.py):
- Target file: [`logs/predictions.jsonl`](file:///c:/Users/Bharath/Downloads/Bank-Fraud-MLOps/logs/predictions.jsonl).
- Thread-safe serialization via `threading.Lock()`.
- Non-blocking error handling: Any logging failure logs a warning and allows inference to complete successfully.
- Record structure includes: `timestamp`, `step`, `amount`, `age`, `gender`, `merchant`, `category`, `prediction`, `fraud_probability`, `risk_level`, `model_version`, and `latency_ms`.

## 18. Runtime Monitoring

Surfaced via `GET /monitoring`:
- Parses live prediction records from `logs/predictions.jsonl`.
- Dynamically aggregates total predictions, fraud vs. legitimate counts, risk distribution tiers, average latency, and average fraud probability.
- Gracefully returns empty structures when log files are absent.

## 19. PSI Data Drift Detection

Surfaced via `GET /monitoring/drift`:
- Compares live production inputs against the reference training baseline distribution (594,643 samples stored in `models/baseline_distribution.json`).
- Applies the Population Stability Index formula with Laplace smoothing ($\epsilon = 10^{-4}$):
  $$\text{PSI} = \sum_{i=1}^{k} (A_i - E_i) \times \ln\left(\frac{A_i}{E_i}\right)$$
- Features evaluated: `amount`, `step`, `age`, `gender`, `category`, `merchant`.
- Thresholds: $\text{PSI} < 0.10$ (`STABLE`), $0.10 \le \text{PSI} \le 0.20$ (`WARNING`), $\text{PSI} > 0.20$ (`DRIFT DETECTED`).
- Enforces minimum sample size: Returns `INSUFFICIENT_DATA` if fewer than 5 transactions are logged.

---

## 20. Docker Containerization

The system provides fully reproducible production container configurations:

- **Backend Container ([`Dockerfile`](file:///c:/Users/Bharath/Downloads/Bank-Fraud-MLOps/Dockerfile)):**
  - Base: `python:3.11-slim`
  - Installs `curl` for container health checks.
  - Leverages layer caching for `requirements.txt`.
  - Exposes port 8000.
  - Health check probes `GET /health` with 3 retries.
  - Starts FastAPI with production Uvicorn (`--host 0.0.0.0 --port 8000`).

- **Frontend Container ([`frontend/Dockerfile`](file:///c:/Users/Bharath/Downloads/Bank-Fraud-MLOps/frontend/Dockerfile)):**
  - Stage 1 (Builder): `node:20-alpine`, runs `npm ci` and `npm run build`.
  - Stage 2 (Server): `nginx:alpine`, serves static assets from `/usr/share/nginx/html`.
  - Configures Single Page Application (SPA) routing fallback in [`frontend/nginx.conf`](file:///c:/Users/Bharath/Downloads/Bank-Fraud-MLOps/frontend/nginx.conf).
  - Health check verifies Nginx web server responses.

- **Orchestration ([`docker-compose.yml`](file:///c:/Users/Bharath/Downloads/Bank-Fraud-MLOps/docker-compose.yml)):**
  - Defines `backend`, `frontend`, and `mlflow` services.
  - Uses Docker bridge network `fraud-net`.
  - Persists prediction logs via named volume `prediction_logs`.
  - Mounts `./mlflow.db` and `./mlruns` for MLflow experiment persistence.

## 21. CI/CD Pipelines

Configured in [`.github/workflows/ci.yml`](file:///c:/Users/Bharath/Downloads/Bank-Fraud-MLOps/.github/workflows/ci.yml):
- **Triggers:** Push or Pull Request to `main` or `master` branches, plus manual dispatch.
- **Backend Job (`backend-test`):**
  - Sets up Python 3.11 with pip caching.
  - Installs dependencies from `requirements.txt`.
  - Verifies model loading and the 79-feature inference schema.
  - Runs all 40 automated tests with pytest.
- **Frontend Job (`frontend-build`):**
  - Sets up Node.js 20 with npm caching.
  - Installs dependencies with `npm ci`.
  - Executes production build with `npm run build`.

---

## 22. Project Structure

```
Bank-Fraud-MLOps/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI workflow
├── app/
│   ├── services/
│   │   ├── __init__.py
│   │   └── prediction_service.py  # FastAPI prediction service layer
│   ├── __init__.py
│   ├── main.py                    # FastAPI application and route endpoints
│   └── schemas.py                 # Pydantic request and response schemas
├── data/                          # Dataset directory (bs140513_032310.csv)
├── frontend/
│   ├── public/                    # Favicon and SVGs
│   ├── src/
│   │   ├── components/            # React UI components (Form, Result, MLOps)
│   │   ├── config/                # Central API endpoint configuration
│   │   ├── App.jsx                # Main application component with tab routing
│   │   ├── index.css              # Dark banking and cybersecurity theme
│   │   └── main.jsx               # React entry point
│   ├── .env.example               # Frontend environment template
│   ├── Dockerfile                 # Multi-stage production Nginx Dockerfile
│   ├── nginx.conf                 # Production Nginx SPA configuration
│   ├── package.json               # Frontend dependencies (React 19, Vite, Lucide)
│   ├── package-lock.json          # Deterministic dependency lockfile
│   └── vite.config.js             # Vite configuration
├── logs/
│   ├── .gitkeep                   # Logs directory keeper
│   └── predictions.jsonl          # Live transaction prediction records
├── mlruns/                        # MLflow local artifact storage
├── models/
│   ├── baseline_distribution.json # Precomputed training quantiles for PSI drift
│   ├── feature_config.json        # 79-feature schema and category mappings
│   ├── model_metadata.json        # Release version and evaluation benchmarks
│   ├── preprocessor.pkl           # Trained preprocessing pipeline
│   └── xgboost_model.pkl          # Trained XGBoost classifier
├── notebooks/                     # Exploratory analysis and modeling notebooks
├── reports/                       # Confusion matrix and performance charts
├── src/
│   ├── mlops/
│   │   ├── __init__.py
│   │   └── mlflow_tracking.py     # MLflow tracking script
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift.py               # Population Stability Index (PSI) drift engine
│   │   ├── metrics.py             # Operational statistics aggregator
│   │   └── prediction_logger.py   # Thread-safe resilient JSONL logger
│   ├── feature_engineering.py     # Feature engineering transformations
│   ├── inference.py               # Production inference interface (79 features)
│   └── preprocessing.py           # Preprocessing pipeline and dummy encoding
├── tests/
│   ├── test_api.py                # 16 FastAPI backend tests
│   ├── test_inference.py          # 15 inference pipeline tests
│   └── test_monitoring.py         # 9 MLOps logging and drift tests
├── .env.example                   # Environment configuration template
├── .gitignore                     # Production Git ignore rules
├── Dockerfile                     # Backend production Dockerfile
├── docker-compose.yml             # Multi-container orchestration
├── mlflow.db                      # Local SQLite tracking database
├── requirements.txt               # Backend Python dependencies
└── README.md                      # Comprehensive project documentation
```

---

## 23. Installation

### Prerequisites
- Python 3.10+ (Tested on Python 3.11 and 3.14)
- Node.js 18+ and npm
- Git

### Clone & Setup Virtual Environment
```bash
git clone https://github.com/vishva08082007/NLP-EXP.git
cd Bank-Fraud-MLOps

# Create and activate Python virtual environment
python -m venv .venv
.venv\Scripts\activate       # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install backend dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 24. Local Development

Copy environment templates:
```bash
cp .env.example .env
cp frontend/.env.example frontend/.env
```

---

## 25. Running MLflow

Initialize and view local experiment tracking:
```bash
# Register model, hyperparameters, and artifacts
python -m src.mlops.mlflow_tracking

# Launch the MLflow Web UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 26. Running Backend

Start the FastAPI server:
```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- Swagger UI Documentation: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**
- Health Check: **[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)**
- Monitoring Telemetry: **[http://127.0.0.1:8000/monitoring](http://127.0.0.1:8000/monitoring)**
- Drift Report: **[http://127.0.0.1:8000/monitoring/drift](http://127.0.0.1:8000/monitoring/drift)**

---

## 27. Running Frontend

In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.

---

## 28. Running with Docker

Ensure Docker Desktop is running, then execute:

```bash
# Validate Docker Compose configuration
docker compose config

# Build all container images
docker compose build

# Launch services in detached mode
docker compose up -d

# Verify container health
docker compose ps

# Access services:
# - Frontend Dashboard: http://localhost:5173
# - FastAPI Documentation: http://localhost:8000/docs
# - MLflow Tracking UI: http://localhost:5000

# Stop services when finished
docker compose down
```

---

## 29. API Documentation

### `POST /predict` (Single Transaction Inference)

#### Request Payload (Legitimate Example)
```json
{
  "step": 41,
  "amount": 6.18,
  "age": "4",
  "gender": "M",
  "merchant": "M1823072687",
  "category": "es_transportation"
}
```

#### Response (`200 OK`)
```json
{
  "prediction": "Legitimate",
  "fraud_probability": 0.0001,
  "risk_level": "LOW",
  "threshold": 0.5,
  "model_version": "1.0.0"
}
```

#### Request Payload (Fraud Example)
```json
{
  "step": 120,
  "amount": 300.58,
  "age": "4",
  "gender": "F",
  "merchant": "M1198415165",
  "category": "es_wellnessandbeauty"
}
```

#### Response (`200 OK`)
```json
{
  "prediction": "Fraud",
  "fraud_probability": 0.9989,
  "risk_level": "HIGH",
  "threshold": 0.5,
  "model_version": "1.0.0"
}
```

### `POST /predict/batch`
```json
{
  "transactions": [
    { "step": 41, "amount": 6.18, "age": "4", "gender": "M", "merchant": "M1823072687", "category": "es_transportation" },
    { "step": 120, "amount": 300.58, "age": "4", "gender": "F", "merchant": "M1198415165", "category": "es_wellnessandbeauty" }
  ]
}
```

### `GET /monitoring`
```json
{
  "total_predictions": 40,
  "fraud_predictions": 12,
  "legitimate_predictions": 28,
  "high_risk_predictions": 12,
  "medium_risk_predictions": 0,
  "low_risk_predictions": 28,
  "average_fraud_probability": 0.3015,
  "average_prediction_latency_ms": 22.92,
  "model_version": "1.0.0",
  "note": "Ground-truth labels are unavailable for current production predictions; model performance metrics shown are evaluation-set metrics."
}
```

### `GET /monitoring/drift`
```json
{
  "overall_status": "DRIFT DETECTED",
  "checked_features": 6,
  "data_available": true,
  "recent_predictions_count": 40,
  "min_required_predictions": null,
  "window_size": 100,
  "message": null,
  "features": [
    {
      "feature": "amount",
      "type": "numerical",
      "method": "Population Stability Index (PSI)",
      "score": 5.0386,
      "threshold": 0.2,
      "status": "DRIFT DETECTED",
      "mean_observed": 83.86,
      "mean_reference": 37.89
    },
    {
      "feature": "gender",
      "type": "categorical",
      "method": "Population Stability Index (PSI)",
      "score": 0.1448,
      "threshold": 0.2,
      "status": "WARNING",
      "mean_observed": null,
      "mean_reference": null
    }
  ]
}
```

---

## 30. Testing & Quality Assurance

Run the automated test suite:
```bash
pytest -v
```

### Test Suite Summary: 40/40 Passing Tests
- **`tests/test_inference.py` (15/15 Passed):** Validates deterministic preprocessing, 79-feature engineering, zero amounts, large amounts, unknown category fallback, and batch consistency.
- **`tests/test_api.py` (16/16 Passed):** Validates `/health`, `/model-info`, `/predict`, `/predict/batch`, Pydantic boundary checks, negative amounts (422), unknown merchants (422), and missing fields (422).
- **`tests/test_monitoring.py` (9/9 Passed):** Validates thread-safe logging, non-blocking failure resilience, live operational counter accuracy, `/monitoring`, minimum sample drift thresholds, PSI score calculations, `/monitoring/drift`, and MLflow experiment runs.

---

## 31. Model Performance

The following are **evaluation-set benchmark metrics** evaluated across 118,929 test records:

- **Accuracy:** `0.9765`
- **Precision:** `0.3370`
- **Recall:** `0.9701` (1,128 of 1,162 frauds identified)
- **F1-Score:** `0.5003`
- **ROC-AUC:** `0.9975`
- **PR-AUC:** `0.8951`
- **Features Required:** 79 engineered features

> **MLOps Notice:** Live production transactions do not include ground-truth fraud labels. Live operational accuracy and recall cannot be computed without verified chargeback or confirmed fraud dispute feeds.

---

## 32. Limitations

1. **Synthetic Base Dataset:** Trained on synthetic transactions representing typical banking behaviors; real-world payment networks encounter wider geographic anomalies.
2. **Delayed Feedback Loop:** Ground-truth fraud labels are typically confirmed 30 to 90 days post-transaction via banking chargeback workflows, creating a delay before retraining data becomes labeled.
3. **Local Store Architecture:** Default MLflow and prediction logs use local filesystem and SQLite storage, which requires shared network storage (e.g. S3 / PostgreSQL) for multi-node horizontal scaling.

---

## 33. Future Improvements

1. **Distributed Cloud Storage:** Connect MLflow and prediction logging to cloud object storage (AWS S3 / Google Cloud Storage) and managed PostgreSQL.
2. **Automated Retraining Pipelines:** Trigger automated model re-training jobs via Airflow or Kubeflow when PSI drift exceeds 0.25 over a 7-day rolling window.
3. **Graph Neural Network (GNN) Features:** Incorporate real-time customer-merchant transaction graph embeddings to capture coordinated fraud rings.
4. **Asynchronous Message Queuing:** Ingest transactions via Apache Kafka or RabbitMQ for decoupled, microsecond ingestion at scale.

---

## 34. Deployment Status

- **Deployment Configuration:** Production configurations (Docker, Docker Compose, Nginx, GitHub Actions CI) are fully implemented and verified locally.
- **Environment Status:** **LOCAL ONLY** (Deployment configuration is prepared, but public cloud deployment was not executed in this phase).
