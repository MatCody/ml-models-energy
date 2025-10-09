# Energy ML API Documentation

## Overview
This API provides machine learning predictions for industrial energy management with three separate endpoints for different models.

## Base URL
```
https://your-space-name.hf.space
```

## Endpoints

### 1. Energy Prediction - Random Forest
**Endpoint:** `/api/predict` (Tab 0)  
**Method:** POST  
**Description:** Predicts daily average energy consumption using Random Forest

#### Request Body
```json
[
  {
    "data": "2025-01-01",
    "boosting": 0.0,
    "cor": "incolor",
    "espessura": 8.0,
    "extracao_forno": 750.0,
    "porcentagem_caco": 15.0,
    "extracao_boosting": 1.5
  }
]
```

#### Response
```json
[
  {
    "data": "01-01-2025",
    "predicted_energy": 5.234,
    "note": "Add boosting energy separately to get final consumption"
  }
]
```

### 2. Energy Prediction - XGBoost
**Endpoint:** `/api/predict` (Tab 1)  
**Method:** POST  
**Description:** Predicts daily average energy consumption using XGBoost (best model)

#### Request Body
```json
[
  {
    "data": "2025-01-01",
    "boosting": 0.0,
    "cor": "incolor",
    "espessura": 8.0,
    "extracao_forno": 750.0,
    "porcentagem_caco": 15.0,
    "extracao_boosting": 1.5
  }
]
```

#### Response
```json
[
  {
    "data": "01-01-2025",
    "predicted_energy": 5.187,
    "note": "Add boosting energy separately to get final consumption"
  }
]
```

### 3. Threshold Detection
**Endpoint:** `/api/predict` (Tab 2)  
**Method:** POST  
**Description:** Predicts probability of exceeding 8300 and 9000 consumption thresholds

#### Request Body
```json
{
  "data": "2025-01-01",
  "cor": "incolor",
  "espessura": 8.0,
  "ext_boosting": 1.5,
  "extracao_forno": 750.0,
  "porcentagem_caco": 15.0,
  "prod_e": 1,
  "prod_l": 0,
  "autoclave": 1
}
```

#### Response
```json
{
  "predictions": {
    "prediction_1": [
      {
        "datetime": "2025-01-01",
        "threshold": 8300,
        "probabilidade_de_estouro": 0.1234,
        "estouro_previsto": 0
      }
    ],
    "prediction_2": [
      {
        "datetime": "2025-01-01",
        "threshold": 9000,
        "probabilidade_de_estouro": 0.0567,
        "estouro_previsto": 0
      }
    ]
  }
}
```

## Input Parameters

### Energy Prediction Models
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | string | Yes | Date in YYYY-MM-DD format |
| `boosting` | float | Yes | Always set to 0.0 (compatibility) |
| `cor` | string | Yes | Glass color: "incolor", "verde", "cinza", "bronze" |
| `espessura` | float | Yes | Glass thickness (mm) |
| `extracao_forno` | float | Yes | Furnace extraction rate |
| `porcentagem_caco` | float | Yes | Glass cullet percentage (%) |
| `extracao_boosting` | float | Yes | External boosting parameter |

### Threshold Detection Model
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | string | Yes | Date in YYYY-MM-DD format |
| `cor` | string | Yes | Glass color: "incolor", "verde", "cinza", "bronze" |
| `espessura` | float | Yes | Glass thickness (mm) |
| `ext_boosting` | float | Yes | External boosting parameter |
| `extracao_forno` | float | Yes | Furnace extraction rate |
| `porcentagem_caco` | float | Yes | Glass cullet percentage (%) |
| `prod_e` | int | Yes | Production E (0 or 1) |
| `prod_l` | int | Yes | Production L (0 or 1) |
| `autoclave` | int | Yes | Autoclave usage (0 or 1) |

## Usage Examples

### Python
```python
import requests
import json

# Energy Prediction (XGBoost)
url = "https://your-space-name.hf.space/api/predict"
headers = {"Content-Type": "application/json"}

data = [
    {
        "data": "2025-01-01",
        "boosting": 0.0,
        "cor": "verde",
        "espessura": 10.0,
        "extracao_forno": 849.0,
        "porcentagem_caco": 45.0,
        "extracao_boosting": 54.0
    }
]

response = requests.post(url, json=data, headers=headers)
prediction = response.json()
print(f"Predicted energy: {prediction[0]['predicted_energy']} MWh")
```

### cURL
```bash
# Threshold Detection
curl -X POST https://your-space-name.hf.space/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": "2025-01-01",
    "cor": "incolor",
    "espessura": 8.0,
    "ext_boosting": 1.5,
    "extracao_forno": 750.0,
    "porcentagem_caco": 15.0,
    "prod_e": 1,
    "prod_l": 0,
    "autoclave": 1
  }'
```

## Model Performance
- **Random Forest Energy**: R² = 0.50, MAE = 0.24 MWh
- **XGBoost Energy**: R² = 0.53, MAE = 0.24 MWh (recommended)
- **Threshold Detection**: 98.9% accuracy (8300), 100% accuracy (9000)

## Notes
- **Energy models predict base consumption** - add actual boosting energy separately
- **No temporal features** - models work for any date/time period
- **Real-time prediction** - sub-second response times
- **Batch processing** - energy models accept arrays of inputs
- **Threshold models** output probabilities (0-1) and binary predictions (>0.5)

## Error Handling
- Invalid JSON: Returns error message
- Missing parameters: Uses default values where possible
- Model errors: Returns error description with context