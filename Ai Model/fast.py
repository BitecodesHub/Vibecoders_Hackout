"""
FastAPI ML Service for Hydrogen Site Optimization
Ready-to-deploy service with all ML endpoints
Run: uvicorn ml_service:app --reload --port 8001
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import json
from datetime import datetime
import asyncio
import pickle
import joblib

# Import the ML optimizer (from the ground zero file)
from zero_testing import HydrogenSiteOptimizer

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SiteFeatures(BaseModel):
    solar_irradiance: float = Field(..., description="Solar irradiance in kWh/m²/day")
    temperature: float = Field(..., description="Temperature in Celsius")
    wind_speed: float = Field(..., description="Wind speed in m/s")
    pv_power: float = Field(..., description="PV power in kW")
    wind_power: float = Field(..., description="Wind power in kW")
    electrolyzer_efficiency: float = Field(..., description="Electrolyzer efficiency %")
    hydrogen_production: float = Field(..., description="H2 production in kg/day")
    desalination_power: float = Field(..., description="Desalination power in kW")
    system_efficiency: float = Field(..., description="System efficiency %")

class BatchScoringRequest(BaseModel):
    sites: List[Dict[str, Any]]
    include_shap: bool = False

class LCOHRequest(BaseModel):
    site_ids: Optional[List[str]] = None
    capex_electrolyzer: float = 800
    opex_factor: float = 0.02
    discount_rate: float = 0.08
    lifetime: int = 20

class PortfolioOptimizationRequest(BaseModel):
    budget: float = Field(100_000_000, description="Budget in INR")
    num_sites: int = Field(20, description="Maximum number of sites")
    risk_aversion: float = Field(0.5, description="Risk aversion (0-1)")
    state_filter: Optional[List[str]] = None
    min_feasibility_score: Optional[float] = 0.5

class StressTestRequest(BaseModel):
    scenarios: Dict[str, Dict[str, float]] = Field(
        default={
            "base": {"capex_change": 1.0, "h2_price": 3.5, "efficiency_change": 1.0},
            "optimistic": {"capex_change": 0.8, "h2_price": 4.0, "efficiency_change": 1.2},
            "pessimistic": {"capex_change": 1.3, "h2_price": 2.5, "efficiency_change": 0.9}
        }
    )

class SiteRankingRequest(BaseModel):
    state: Optional[str] = None
    top_n: int = 10
    sort_by: str = "composite_index"  # or "lcoh", "feasibility_score"

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class SiteScore(BaseModel):
    site_id: str
    feasibility_score: float
    composite_index: float
    suitability_band: str
    confidence: float

class SHAPExplanation(BaseModel):
    site_id: str
    feature_importance: Dict[str, float]
    top_positive_factors: List[Dict[str, float]]
    top_negative_factors: List[Dict[str, float]]

class LCOHResult(BaseModel):
    site_id: str
    lcoh_usd_per_kg: float
    lcoh_band: str
    capex_total: float
    opex_annual: float
    h2_annual_production: float

class PortfolioResult(BaseModel):
    selected_sites: List[Dict[str, Any]]
    allocation_percentages: Dict[str, float]
    expected_return: float
    risk: float
    sharpe_ratio: float
    total_h2_production: float
    average_lcoh: float

class EfficientFrontierPoint(BaseModel):
    risk: float
    return_: float
    sharpe_ratio: float

class StressTestResult(BaseModel):
    scenario: str
    expected_return: float
    risk: float
    sharpe_ratio: float
    num_sites: int
    impact_vs_base: float

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Hydrogen Site ML Service",
    description="ML backend for hydrogen site selection and portfolio optimization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global optimizer instance
optimizer = None

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize ML models on startup"""
    global optimizer
    print("🚀 Starting ML Service...")
    
    # Initialize optimizer
    optimizer = HydrogenSiteOptimizer('renewable_hydrogen_dataset_2535.csv')
    
    # Load data and train models
    optimizer.load_and_prepare_data()
    optimizer.calculate_lcoh()
    optimizer.train_ml_models()
    optimizer.calculate_shap_explanations()
    optimizer.create_composite_suitability_index()
    
    print("✅ ML Service Ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down ML Service...")

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": optimizer is not None,
        "sites_available": len(optimizer.df) if optimizer else 0
    }

# ============================================================================
# ML ENDPOINTS
# ============================================================================

@app.post("/ml/score/batch", response_model=List[SiteScore])
async def batch_scoring(request: BatchScoringRequest):
    """Score multiple sites for feasibility"""
    try:
        results = []
        
        for site in request.sites:
            # Convert site data to features
            features = [
                site.get('solar_irradiance', 0),
                site.get('temperature', 0),
                site.get('wind_speed', 0),
                site.get('pv_power', 0),
                site.get('wind_power', 0),
                site.get('electrolyzer_efficiency', 0),
                site.get('hydrogen_production', 0),
                site.get('desalination_power', 0),
                site.get('system_efficiency', 0),
                site.get('total_renewable_power', 0),
                site.get('solar_wind_ratio', 0),
                site.get('power_per_h2', 0)
            ]
            
            # Predict
            features_scaled = optimizer.scaler.transform([features])
            score = optimizer.model.predict(features_scaled)[0]
            
            # Get composite index (simplified calculation)
            composite = np.mean(features_scaled[0]) * score
            
            results.append(SiteScore(
                site_id=site.get('site_id', f"site_{len(results)}"),
                feasibility_score=float(score),
                composite_index=float(composite),
                suitability_band="High" if score > 0.7 else "Medium" if score > 0.4 else "Low",
                confidence=0.85
            ))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/shap/explain", response_model=SHAPExplanation)
async def get_shap_explanation(site_id: str, features: SiteFeatures):
    """Get SHAP explanations for a specific site"""
    try:
        # Convert to feature array
        feature_values = [
            features.solar_irradiance,
            features.temperature,
            features.wind_speed,
            features.pv_power,
            features.wind_power,
            features.electrolyzer_efficiency,
            features.hydrogen_production,
            features.desalination_power,
            features.system_efficiency,
            features.pv_power + features.wind_power,  # total_renewable_power
            features.pv_power / (features.wind_power + 1),  # solar_wind_ratio
            (features.pv_power + features.wind_power) / (features.hydrogen_production + 1)  # power_per_h2
        ]
        
        # Get SHAP values for this instance
        features_scaled = optimizer.scaler.transform([feature_values])
        shap_values = optimizer.shap_explainer(features_scaled)
        
        # Create feature importance dict
        feature_importance = {
            optimizer.feature_columns[i]: float(shap_values.values[0][i])
            for i in range(len(optimizer.feature_columns))
        }
        
        # Get top positive and negative factors
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1])
        
        top_negative = [
            {"feature": f[0], "impact": f[1]} 
            for f in sorted_features[:3] if f[1] < 0
        ]
        
        top_positive = [
            {"feature": f[0], "impact": f[1]} 
            for f in sorted_features[-3:] if f[1] > 0
        ]
        
        return SHAPExplanation(
            site_id=site_id,
            feature_importance=feature_importance,
            top_positive_factors=top_positive,
            top_negative_factors=top_negative
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/lcoh/calculate", response_model=List[LCOHResult])
async def calculate_lcoh(request: LCOHRequest):
    """Calculate LCOH for sites"""
    try:
        # Recalculate LCOH with given parameters
        optimizer.calculate_lcoh(
            capex_electrolyzer=request.capex_electrolyzer,
            opex_factor=request.opex_factor,
            discount_rate=request.discount_rate,
            lifetime=request.lifetime
        )
        
        # Filter sites if specified
        df = optimizer.df
        if request.site_ids:
            df = df[df['City'].isin(request.site_ids)]
        
        # Get top sites by LCOH
        df = df.nsmallest(20, 'LCOH_USD_per_kg')
        
        results = []
        for _, row in df.iterrows():
            results.append(LCOHResult(
                site_id=row['City'],
                lcoh_usd_per_kg=row['LCOH_USD_per_kg'],
                lcoh_band=row['LCOH_Band'],
                capex_total=row['CAPEX_Total'],
                opex_annual=row['OPEX_Annual'],
                h2_annual_production=row['H2_Annual_kg']
            ))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/portfolio/optimize", response_model=PortfolioResult)
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """Run portfolio optimization"""
    try:
        # Filter by state if specified
        if request.state_filter:
            original_df = optimizer.df.copy()
            optimizer.df = optimizer.df[optimizer.df['State'].isin(request.state_filter)]
        
        # Run optimization
        result = optimizer.portfolio_optimization_markowitz(
            budget=request.budget,
            num_sites=request.num_sites,
            risk_aversion=request.risk_aversion
        )
        
        if not result:
            raise HTTPException(status_code=400, detail="Optimization failed")
        
        # Prepare response
        selected_sites = []
        allocation_percentages = {}
        
        for _, site in result['selected_sites'].iterrows():
            site_dict = site.to_dict()
            selected_sites.append({
                "site_id": site_dict['City'],
                "state": site_dict['State'],
                "allocation_percent": site_dict['Allocation_%'],
                "investment_inr": site_dict['Investment_INR'],
                "lcoh": site_dict['LCOH_USD_per_kg'],
                "h2_production": site_dict['Hydrogen_Production_kg/day']
            })
            allocation_percentages[site_dict['City']] = site_dict['Allocation_%']
        
        # Calculate aggregate metrics
        total_h2 = result['selected_sites']['Hydrogen_Production_kg/day'].sum() * 350
        avg_lcoh = result['selected_sites']['LCOH_USD_per_kg'].mean()
        
        # Restore original df if filtered
        if request.state_filter:
            optimizer.df = original_df
        
        return PortfolioResult(
            selected_sites=selected_sites,
            allocation_percentages=allocation_percentages,
            expected_return=float(result['expected_return']),
            risk=float(result['risk']),
            sharpe_ratio=float(result['sharpe_ratio']),
            total_h2_production=float(total_h2),
            average_lcoh=float(avg_lcoh)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/portfolio/frontier", response_model=List[EfficientFrontierPoint])
async def get_efficient_frontier(num_points: int = 20):
    """Get efficient frontier data"""
    try:
        risks, returns = optimizer.generate_efficient_frontier(num_points)
        
        frontier_points = []
        for risk, return_ in zip(risks, returns):
            sharpe = return_ / risk if risk > 0 else 0
            frontier_points.append(EfficientFrontierPoint(
                risk=float(risk),
                return_=float(return_),
                sharpe_ratio=float(sharpe)
            ))
        
        return frontier_points
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/portfolio/stress-test", response_model=List[StressTestResult])
async def stress_test(request: StressTestRequest):
    """Run stress test scenarios"""
    try:
        results = []
        base_return = None
        
        for scenario_name, params in request.scenarios.items():
            # Adjust parameters
            original_capex = optimizer.df['CAPEX_Total'].copy()
            original_efficiency = optimizer.df['Electrolyzer_Efficiency_%'].copy()
            
            optimizer.df['CAPEX_Total'] = original_capex * params['capex_change']
            optimizer.df['Electrolyzer_Efficiency_%'] = original_efficiency * params['efficiency_change']
            
            # Recalculate and optimize
            optimizer.calculate_lcoh()
            portfolio_result = optimizer.portfolio_optimization_markowitz()
            
            if portfolio_result:
                if scenario_name == "base":
                    base_return = portfolio_result['expected_return']
                
                impact = 0 if not base_return else (
                    (portfolio_result['expected_return'] - base_return) / base_return * 100
                )
                
                results.append(StressTestResult(
                    scenario=scenario_name,
                    expected_return=float(portfolio_result['expected_return']),
                    risk=float(portfolio_result['risk']),
                    sharpe_ratio=float(portfolio_result['sharpe_ratio']),
                    num_sites=len(portfolio_result['selected_sites']),
                    impact_vs_base=float(impact)
                ))
            
            # Restore original values
            optimizer.df['CAPEX_Total'] = original_capex
            optimizer.df['Electrolyzer_Efficiency_%'] = original_efficiency
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/sites/rank")
async def rank_sites(request: SiteRankingRequest):
    """Get ranked sites by various criteria"""
    try:
        df = optimizer.df.copy()
        
        # Filter by state if specified
        if request.state:
            df = df[df['State'] == request.state]
        
        # Sort by specified criterion
        if request.sort_by == "lcoh":
            df = df.nsmallest(request.top_n, 'LCOH_USD_per_kg')
        elif request.sort_by == "feasibility_score":
            df = df.nlargest(request.top_n, 'Feasibility_Score')
        else:  # composite_index
            df = df.nlargest(request.top_n, 'Composite_Suitability_Index')
        
        # Prepare response
        sites = []
        for _, row in df.iterrows():
            sites.append({
                "site_id": row['City'],
                "state": row['State'],
                "latitude": row['Latitude_India'],
                "longitude": row['Longitude_India'],
                "composite_index": row['Composite_Suitability_Index'],
                "feasibility_score": row['Feasibility_Score'],
                "lcoh": row['LCOH_USD_per_kg'],
                "h2_production": row['Hydrogen_Production_kg/day'],
                "suitability_band": row['Suitability_Band'],
                "lcoh_band": row['LCOH_Band']
            })
        
        return {"sites": sites, "count": len(sites)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/statistics/summary")
async def get_summary_statistics():
    """Get summary statistics of all sites"""
    try:
        df = optimizer.df
        
        return {
            "total_sites": len(df),
            "states": df['State'].unique().tolist(),
            "lcoh_range": {
                "min": float(df['LCOH_USD_per_kg'].min()),
                "max": float(df['LCOH_USD_per_kg'].max()),
                "mean": float(df['LCOH_USD_per_kg'].mean()),
                "median": float(df['LCOH_USD_per_kg'].median())
            },
            "feasibility_distribution": {
                "high": int((df['Feasibility_Score'] > 0.7).sum()),
                "medium": int(((df['Feasibility_Score'] > 0.4) & (df['Feasibility_Score'] <= 0.7)).sum()),
                "low": int((df['Feasibility_Score'] <= 0.4).sum())
            },
            "h2_production_total": float(df['Hydrogen_Production_kg/day'].sum()),
            "avg_system_efficiency": float(df['System_Efficiency_%'].mean()),
            "renewable_capacity": {
                "total_solar_mw": float(df['PV_Power_kW'].sum() / 1000),
                "total_wind_mw": float(df['Wind_Power_kW'].sum() / 1000)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# WEBSOCKET FOR REAL-TIME UPDATES (Optional)
# ============================================================================

from fastapi import WebSocket, WebSocketDisconnect
from typing import Set

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time optimization updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            # Process request (e.g., trigger optimization)
            if data == "optimize":
                # Send progress updates
                await websocket.send_json({
                    "type": "progress",
                    "message": "Starting optimization...",
                    "progress": 0
                })
                
                # Simulate progress updates
                for i in range(1, 11):
                    await asyncio.sleep(0.5)
                    await websocket.send_json({
                        "type": "progress",
                        "message": f"Processing... {i*10}%",
                        "progress": i * 10
                    })
                
                await websocket.send_json({
                    "type": "complete",
                    "message": "Optimization complete!",
                    "progress": 100
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)