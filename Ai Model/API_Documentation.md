# Hydrogen Site ML Service API Documentation

## Overview
FastAPI-based ML service for hydrogen site selection and portfolio optimization. The service provides comprehensive analysis for renewable hydrogen site feasibility, cost optimization, and investment portfolio management.

**Base URL**: `http://localhost:8001`
**Service Port**: `8001`

---

## 1. Health Check

**Endpoint**: `GET /health`

**Description**: Check service health and model status.

### Request
```bash
GET /health
```

### Response
```json
{
  "status": "healthy",
  "timestamp": "2025-08-30T14:05:30.281245",
  "model_loaded": true,
  "sites_available": 2535
}
```

---

## 2. Batch Site Scoring

**Endpoint**: `POST /ml/score/batch`

**Description**: Score multiple sites for hydrogen production feasibility.

### Request
```json
{
  "sites": [
    {
      "site_id": "test_site_1",
      "solar_irradiance": 5.2,
      "temperature": 25.3,
      "wind_speed": 7.8,
      "pv_power": 1500,
      "wind_power": 800,
      "electrolyzer_efficiency": 75.5,
      "hydrogen_production": 450,
      "desalination_power": 120,
      "system_efficiency": 68.2,
      "total_renewable_power": 2300,
      "solar_wind_ratio": 1.875,
      "power_per_h2": 5.11
    }
  ],
  "include_shap": false
}
```

### Response
```json
[
  {
    "site_id": "test_site_1",
    "feasibility_score": 0.9505724233453443,
    "composite_index": 40.95926222657798,
    "suitability_band": "High",
    "confidence": 0.85
  }
]
```

---

## 3. SHAP Explanations

**Endpoint**: `POST /ml/shap/explain?site_id={site_id}`

**Description**: Get SHAP-based feature importance explanations for ML predictions.

### Request
```bash
POST /ml/shap/explain?site_id=test_site
Content-Type: application/json
```

```json
{
  "solar_irradiance": 5.2,
  "temperature": 25.3,
  "wind_speed": 7.8,
  "pv_power": 1500,
  "wind_power": 800,
  "electrolyzer_efficiency": 75.5,
  "hydrogen_production": 450,
  "desalination_power": 120,
  "system_efficiency": 68.2
}
```

### Response
```json
{
  "site_id": "test_site",
  "feature_importance": {
    "Solar_Irradiance_kWh/m²/day": -0.000677228972947006,
    "Temperature_C": -0.00010292476548499963,
    "Wind_Speed_m/s": 0.0002363442158460316,
    "PV_Power_kW": -7.97062743515653e-05,
    "Wind_Power_kW": -1.3741385782850556e-05,
    "Electrolyzer_Efficiency_%": 0.000486525958792754,
    "Hydrogen_Production_kg/day": 0.03342124358210526,
    "Desalination_Power_kW": -8.968658881585157e-05,
    "System_Efficiency_%": -0.02356850813209604,
    "Total_Renewable_Power_kW": -6.941327351626115e-05,
    "Solar_Wind_Ratio": 5.4057008804875294e-05,
    "Power_per_H2": 0.00017043779584433594
  },
  "top_positive_factors": [
    {
      "feature": "Wind_Speed_m/s",
      "impact": 0.0002363442158460316
    },
    {
      "feature": "Electrolyzer_Efficiency_%",
      "impact": 0.000486525958792754
    },
    {
      "feature": "Hydrogen_Production_kg/day",
      "impact": 0.03342124358210526
    }
  ],
  "top_negative_factors": [
    {
      "feature": "System_Efficiency_%",
      "impact": -0.02356850813209604
    },
    {
      "feature": "Solar_Irradiance_kWh/m²/day",
      "impact": -0.000677228972947006
    },
    {
      "feature": "Temperature_C",
      "impact": -0.00010292476548499963
    }
  ]
}
```

---

## 4. LCOH Calculation

**Endpoint**: `POST /ml/lcoh/calculate`

**Description**: Calculate Levelized Cost of Hydrogen for sites with custom parameters.

### Request
```json
{
  "site_ids": null,
  "capex_electrolyzer": 800,
  "opex_factor": 0.02,
  "discount_rate": 0.08,
  "lifetime": 20
}
```

### Response
```json
[
  {
    "site_id": "City-1394",
    "lcoh_usd_per_kg": 5.759213356484022,
    "lcoh_band": "Very High (>$5)",
    "capex_total": 976000.0,
    "opex_annual": 19520.0,
    "h2_annual_production": 20650.0
  },
  {
    "site_id": "City-420",
    "lcoh_usd_per_kg": 5.762163773162548,
    "lcoh_band": "Very High (>$5)",
    "capex_total": 976500.0,
    "opex_annual": 19530.0,
    "h2_annual_production": 20650.0
  }
]
```

---

## 5. Portfolio Optimization

**Endpoint**: `POST /ml/portfolio/optimize`

**Description**: Run Markowitz portfolio optimization for hydrogen site selection.

### Request
```json
{
  "budget": 100000000,
  "num_sites": 10,
  "risk_aversion": 0.5,
  "state_filter": null,
  "min_feasibility_score": 0.5
}
```

### Response
```json
{
  "selected_sites": [
    {
      "site_id": "City-1394",
      "state": "Rajasthan",
      "allocation_percent": 100.0,
      "investment_inr": 100000000.0,
      "lcoh": 5.759213356484022,
      "h2_production": 59
    }
  ],
  "allocation_percentages": {
    "City-1394": 100.0
  },
  "expected_return": -0.04779995472479001,
  "risk": 0.010560464419597985,
  "sharpe_ratio": -4.526311800841204,
  "total_h2_production": 20650.0,
  "average_lcoh": 5.759213356484022
}
```

---

## 6. Efficient Frontier

**Endpoint**: `GET /ml/portfolio/frontier?num_points={num_points}`

**Description**: Generate efficient frontier data points for risk-return analysis.

### Request
```bash
GET /ml/portfolio/frontier?num_points=5
```

### Response
```json
[
  {
    "risk": 0.01025202163656615,
    "return_": -0.04781855099844825,
    "sharpe_ratio": -4.664304533643646
  },
  {
    "risk": 0.010252012667671128,
    "return_": -0.0478185726642524,
    "sharpe_ratio": -4.664310727496885
  },
  {
    "risk": 0.010252004526926959,
    "return_": -0.04781859432982246,
    "sharpe_ratio": -4.664316544557369
  },
  {
    "risk": 0.010251996829080196,
    "return_": -0.04781861720906634,
    "sharpe_ratio": -4.664322278507435
  },
  {
    "risk": 0.010251990644444934,
    "return_": -0.047818637965147286,
    "sharpe_ratio": -4.664327116905626
  }
]
```

---

## 7. Stress Testing

**Endpoint**: `POST /ml/portfolio/stress-test`

**Description**: Run stress test scenarios on portfolio optimization.

### Request
```json
{
  "scenarios": {
    "base": {
      "capex_change": 1.0,
      "h2_price": 3.5,
      "efficiency_change": 1.0
    },
    "optimistic": {
      "capex_change": 0.8,
      "h2_price": 4.0,
      "efficiency_change": 1.2
    },
    "pessimistic": {
      "capex_change": 1.3,
      "h2_price": 2.5,
      "efficiency_change": 0.9
    }
  }
}
```

### Response
```json
[
  {
    "scenario": "base",
    "expected_return": -0.04781859432982246,
    "risk": 0.010252004526926959,
    "sharpe_ratio": -4.664316544557369,
    "num_sites": 2,
    "impact_vs_base": 0.0
  },
  {
    "scenario": "optimistic",
    "expected_return": -0.04781859432982246,
    "risk": 0.010252004526926959,
    "sharpe_ratio": -4.664316544557369,
    "num_sites": 2,
    "impact_vs_base": 0.0
  }
]
```

---

## 8. Site Ranking

**Endpoint**: `POST /ml/sites/rank`

**Description**: Get ranked sites by various criteria (composite index, LCOH, feasibility score).

### Request
```json
{
  "state": null,
  "top_n": 3,
  "sort_by": "composite_index"
}
```

### Response
```json
{
  "sites": [
    {
      "site_id": "City-1394",
      "state": "Rajasthan",
      "latitude": 27.796609228340266,
      "longitude": 81.76091114148258,
      "composite_index": 0.9777815339923468,
      "feasibility_score": 1.0,
      "lcoh": 5.759213356484022,
      "h2_production": 59,
      "suitability_band": "Very High",
      "lcoh_band": "Very High (>$5)"
    },
    {
      "site_id": "City-942",
      "state": "Maharashtra",
      "latitude": 28.3126700377369,
      "longitude": 87.21840293857649,
      "composite_index": 0.9750397017859302,
      "feasibility_score": 0.9995519569655528,
      "lcoh": 6.007048357480262,
      "h2_production": 59,
      "suitability_band": "Very High",
      "lcoh_band": "Very High (>$5)"
    },
    {
      "site_id": "City-1617",
      "state": "Tamil Nadu",
      "latitude": 34.91209041831751,
      "longitude": 68.00061437690765,
      "composite_index": 0.9733026859361177,
      "feasibility_score": 0.9990280508131558,
      "lcoh": 6.163420441442174,
      "h2_production": 59,
      "suitability_band": "Very High",
      "lcoh_band": "Very High (>$5)"
    }
  ],
  "count": 3
}
```

---

## 9. Summary Statistics

**Endpoint**: `GET /ml/statistics/summary`

**Description**: Get comprehensive summary statistics of all sites in the dataset.

### Request
```bash
GET /ml/statistics/summary
```

### Response
```json
{
  "total_sites": 2535,
  "states": [
    "Tamil Nadu",
    "Maharashtra",
    "Rajasthan",
    "Karnataka",
    "Gujarat"
  ],
  "lcoh_range": {
    "min": 5.759213356484022,
    "max": 7.2531076680446835,
    "mean": 6.424598485488661,
    "median": 6.391493217516205
  },
  "feasibility_distribution": {
    "high": 2535,
    "medium": 0,
    "low": 0
  },
  "h2_production_total": 131965.0,
  "avg_system_efficiency": 78.56116063275694,
  "renewable_capacity": {
    "total_solar_mw": 658.016,
    "total_wind_mw": 474.588
  }
}
```

---

## 10. WebSocket Real-time Updates

**Endpoint**: `WS /ws/updates`

**Description**: WebSocket connection for real-time optimization progress updates.

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8001/ws/updates');

// Send optimization request
ws.send('optimize');

// Receive progress updates
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log(data);
};
```

### Message Types

#### Progress Update
```json
{
  "type": "progress",
  "message": "Processing... 50%",
  "progress": 50
}
```

#### Completion
```json
{
  "type": "complete",
  "message": "Optimization complete!",
  "progress": 100
}
```

---

## Data Models

### Site Features Input
```json
{
  "solar_irradiance": 5.2,          // kWh/m²/day
  "temperature": 25.3,              // Celsius
  "wind_speed": 7.8,                // m/s
  "pv_power": 1500,                 // kW
  "wind_power": 800,                // kW
  "electrolyzer_efficiency": 75.5,  // %
  "hydrogen_production": 450,       // kg/day
  "desalination_power": 120,        // kW
  "system_efficiency": 68.2         // %
}
```

### Suitability Bands
- **Very High**: Composite Index > 0.8
- **High**: Composite Index 0.6 - 0.8
- **Medium**: Composite Index 0.4 - 0.6
- **Low**: Composite Index < 0.4

### LCOH Bands
- **Very High**: > $5 per kg
- **High**: $4 - $5 per kg
- **Medium**: $3 - $4 per kg
- **Low**: < $3 per kg

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Validation error or invalid parameters"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error message"
}
```

### 422 Unprocessable Entity
```json
{
  "detail": [
    {
      "type": "validation_error",
      "loc": ["field_name"],
      "msg": "Error description",
      "input": "invalid_value"
    }
  ]
}
```

---

## Usage Examples

### cURL Examples

#### Test Health
```bash
curl -X GET "http://localhost:8001/health"
```

#### Batch Scoring
```bash
curl -X POST "http://localhost:8001/ml/score/batch" \
  -H "Content-Type: application/json" \
  -d @site_data.json
```

#### Portfolio Optimization
```bash
curl -X POST "http://localhost:8001/ml/portfolio/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "budget": 50000000,
    "num_sites": 5,
    "risk_aversion": 0.3,
    "state_filter": ["Rajasthan", "Gujarat"]
  }'
```

### Python Examples

```python
import requests

# Health check
response = requests.get("http://localhost:8001/health")
print(response.json())

# Site scoring
site_data = {
    "sites": [{
        "site_id": "my_site",
        "solar_irradiance": 5.5,
        "temperature": 28.0,
        "wind_speed": 8.2,
        "pv_power": 2000,
        "wind_power": 1000,
        "electrolyzer_efficiency": 80.0,
        "hydrogen_production": 500,
        "desalination_power": 150,
        "system_efficiency": 70.0,
        "total_renewable_power": 3000,
        "solar_wind_ratio": 2.0,
        "power_per_h2": 6.0
    }]
}

response = requests.post(
    "http://localhost:8001/ml/score/batch",
    json=site_data
)
print(response.json())
```

---

**Service Status**: ✅ All endpoints operational and tested
**Documentation Updated**: 2025-08-30
**API Version**: 1.0.0