# IoT Predictive Maintenance System | Edge AI & Distributed Sensing

An end-to-end machine learning pipeline for real-time predictive maintenance of industrial IoT equipment, deployed as a REST API with Docker containerisation.

## Problem Statement

Industrial IoT systems generate continuous sensor data from rotating machinery. Unplanned equipment failures cause significant downtime and cost. This system uses supervised ML to detect failure patterns **before they occur**, enabling proactive maintenance.

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — UCI Machine Learning Repository
- 10,000 real-world industrial sensor readings
- 5 raw sensor features: Air Temperature, Process Temperature, Rotational Speed, Torque, Tool Wear
- Binary label: Machine Failure (3.39% failure rate — highly imbalanced)

## Model & Results

| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** *(Best)* | **99.05%** | **85.27%** | **90.16%** | **80.88%** | **97.14%** |
| Isolation Forest *(Unsupervised Baseline)* | 95.36% | 31.56% | — | — | — |

**5-Fold Cross-Validation F1: 83.96% (±2.27%)**

> The Isolation Forest baseline confirms the challenge of unsupervised anomaly detection on heavily imbalanced data. The supervised Random Forest with `class_weight='balanced'` significantly outperforms it, achieving production-grade reliability.

## Feature Engineering

Beyond the 5 raw sensor readings, 4 domain-knowledge features were engineered:

| Feature | Formula | Rationale |
|---|---|---|
| `temp_diff` | Process Temp − Air Temp | Thermal stress indicator |
| `power` | RPM × Torque | Mechanical power output |
| `wear_torque` | Tool Wear × Torque | Combined degradation signal |
| `speed_torque_ratio` | RPM / (Torque + 1) | Operational efficiency metric |

## Technical Architecture

```
IoT Sensor Node
     │
     ▼
FastAPI REST API (/predict)
     │
     ▼
StandardScaler → Feature Engineering → Random Forest
     │
     ▼
Prediction: NORMAL / FAILURE + Probability + Risk Level
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service info and model metrics |
| `/health` | GET | Health check |
| `/predict` | POST | Real-time failure prediction |
| `/docs` | GET | Interactive Swagger UI |

### Sample Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "air_temperature_K": 298.1,
    "process_temperature_K": 308.6,
    "rotational_speed_rpm": 1551.0,
    "torque_Nm": 42.8,
    "tool_wear_min": 0.0
  }'
```

### Sample Response

```json
{
  "prediction": "NORMAL",
  "failure_probability": 0.0312,
  "risk_level": "LOW",
  "confidence": "High confidence — normal operation",
  "features_used": {
    "air_temperature_K": 298.1,
    "process_temperature_K": 308.6,
    "temp_diff": 10.5,
    "power": 66381.8,
    "wear_torque": 0.0
  },
  "model_info": {
    "model": "Random Forest",
    "f1_score": "85.27%",
    "roc_auc": "97.14%"
  }
}
```

## Setup & Run

### Local
```bash
git clone https://github.com/mannat2376/IoT-Anomaly-Detection.git
cd IoT-Anomaly-Detection
pip install -r requirements.txt

# Train the model first
python notebooks/supervised_detection.py

# Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t iot-maintenance-api .
docker run -p 8000:8000 iot-maintenance-api
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

## Project Structure

```
IoT-Anomaly-Detection/
├── api/
│   └── main.py              # FastAPI REST endpoint
├── data/
│   └── sensor_data.csv      # AI4I 2020 dataset (UCI)
├── models/
│   ├── random_forest.pkl    # Trained model
│   ├── scaler.pkl           # StandardScaler
│   └── metrics.txt          # Verified evaluation report
├── notebooks/
│   ├── supervised_detection.py   # Full ML pipeline
│   └── anomaly_detection.py      # Unsupervised baseline
├── Dockerfile
├── requirements.txt
└── README.md
```

## Contributors

- **Mannat Singh** — ML Pipeline, Feature Engineering, FastAPI Deployment, Docker
