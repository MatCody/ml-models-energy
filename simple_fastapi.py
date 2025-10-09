#!/usr/bin/env python3
"""
FastAPI Simplificada para Predição de Energia
Retorna JSON direto sem streaming do Gradio
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'

app = FastAPI(
    title="Energy Prediction API",
    description="API para predição de energia e detecção de limiares",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class EnergyInput(BaseModel):
    data: str  # YYYY-MM-DD format
    boosting: float = 0.0
    cor: str = "incolor"
    espessura: float = 8.0
    extracao_forno: float = 750.0
    porcentagem_caco: float = 15.0
    extracao_boosting: float = 1.5

class ThresholdInput(BaseModel):
    data: str  # YYYY-MM-DD format
    cor: str = "incolor"
    espessura: float = 8.0
    extracao_forno: float = 750.0
    porcentagem_caco: float = 15.0
    ext_boosting: float = 1.5
    prod_e: int = 1
    prod_l: int = 0
    autoclave: int = 1

class EnergyOutput(BaseModel):
    data: str  # DD-MM-YYYY format
    predicted_energy: float

class ThresholdOutput(BaseModel):
    datetime: str  # DD-MM-YYYY format
    threshold: int
    probabilidade_de_estouro: float
    estouro_previsto: int

class ThresholdResponse(BaseModel):
    predictions: Dict[str, List[ThresholdOutput]]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    timestamp: str

class SimpleEnergyPredictor:
    def __init__(self):
        self.rf_model = None
        self.rf_preprocessor = None
        self.xgb_model = None
        self.threshold_model_83 = None
        self.threshold_model_90 = None
        self.models_loaded = False
        
    def load_models(self):
        """Load all models"""
        try:
            # Load RF Energy Model
            if os.path.exists('GRADIO/rf_energy_model.pkl'):
                with open('GRADIO/rf_energy_model.pkl', 'rb') as f:
                    rf_data = pickle.load(f)
                    self.rf_model = rf_data['model']
                    self.rf_preprocessor = rf_data['preprocessor']
                print("✅ Loaded RF energy model")
            
            # Load XGBoost as fallback
            self.xgb_model = self.rf_model
            
            # Load Threshold Models
            if os.path.exists('GRADIO/xgboost_threshold_8300_model.pkl'):
                with open('GRADIO/xgboost_threshold_8300_model.pkl', 'rb') as f:
                    self.threshold_model_83 = pickle.load(f)['model']
                print("✅ Loaded threshold 8300 model")
            
            if os.path.exists('GRADIO/xgboost_threshold_9000_model.pkl'):
                with open('GRADIO/xgboost_threshold_9000_model.pkl', 'rb') as f:
                    self.threshold_model_90 = pickle.load(f)['model']
                print("✅ Loaded threshold 9000 model")
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

# Initialize predictor
predictor = SimpleEnergyPredictor()

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    success = predictor.load_models()
    if not success:
        print("⚠️ Warning: Some models failed to load")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=predictor.models_loaded,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict/energy", response_model=List[EnergyOutput])
async def predict_energy(inputs: List[EnergyInput]):
    """
    Simple energy prediction endpoint
    Input: [{"data": "2025-01-01", "boosting": 0.0, "cor": "incolor", "espessura": 8.0, "extracao_forno": 750.0, "porcentagem_caco": 15.0, "extracao_boosting": 1.5}]
    Output: [{"data": "01-01-2025", "predicted_energy": 4.812}]
    """
    if not predictor.models_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        results = []
        
        for item in inputs:
            # Parse date
            date_obj = datetime.strptime(item.data, '%Y-%m-%d')
            
            # Get features
            extracao_forno_val = float(str(item.extracao_forno).replace(',', '.'))
            
            # Create input
            input_data = {
                'boosting': 0.0,  # Always 0 for compatibility
                'cor': str(item.cor).lower(),
                'espessura': float(item.espessura),
                'extracao_forno': extracao_forno_val,
                'porcentagem_caco': float(item.porcentagem_caco),
                'extracao_boosting': float(item.extracao_boosting)
            }
            
            # Predict
            input_df = pd.DataFrame([input_data])
            X_processed = predictor.rf_preprocessor.transform(input_df)
            prediction = predictor.rf_model.predict(X_processed)[0]
            
            results.append(EnergyOutput(
                data=date_obj.strftime('%d-%m-%Y'),  # DD-MM-YYYY format
                predicted_energy=round(float(prediction), 6)
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/threshold", response_model=ThresholdResponse)
async def predict_threshold(inputs: List[ThresholdInput]):
    """
    Threshold prediction endpoint
    Input: [{"data": "2025-01-01", "cor": "incolor", "espessura": 8.0, "extracao_forno": 750.0, "porcentagem_caco": 15.0, "ext_boosting": 1.5, "prod_e": 1, "prod_l": 0, "autoclave": 1}]
    Output: {"predictions": {"prediction_1": [8300 results], "prediction_2": [9000 results]}}
    """
    if not predictor.models_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        results_83 = []
        results_90 = []
        
        for item in inputs:
            # Parse date
            date_obj = datetime.strptime(item.data, '%Y-%m-%d')
            
            # Encode categorical
            cor_mapping = {'incolor': 0, 'verde': 1, 'cinza': 2, 'bronze': 3}
            cor_encoded = cor_mapping.get(str(item.cor).lower(), 0)
            
            # Create features array
            features = np.array([[
                float(item.espessura),
                float(str(item.extracao_forno).replace(',', '.')),
                float(item.porcentagem_caco),
                float(item.ext_boosting),
                cor_encoded,
                int(item.prod_e),
                int(item.prod_l),
                int(item.autoclave)
            ]])
            
            # Predict probabilities
            try:
                prob_83 = predictor.threshold_model_83.predict_proba(features)[0][1] if len(predictor.threshold_model_83.classes_) > 1 else 0.0
            except:
                prob_83 = 0.0
                
            try:
                prob_90 = predictor.threshold_model_90.predict_proba(features)[0][1] if len(predictor.threshold_model_90.classes_) > 1 else 0.0
            except:
                prob_90 = 0.0
            
            # Add results
            results_83.append(ThresholdOutput(
                datetime=date_obj.strftime('%d-%m-%Y'),  # DD-MM-YYYY format
                threshold=8300,
                probabilidade_de_estouro=round(prob_83, 4),
                estouro_previsto=int(prob_83 > 0.5)
            ))
            
            results_90.append(ThresholdOutput(
                datetime=date_obj.strftime('%d-%m-%Y'),  # DD-MM-YYYY format
                threshold=9000,
                probabilidade_de_estouro=round(prob_90, 4),
                estouro_previsto=int(prob_90 > 0.5)
            ))
        
        return ThresholdResponse(
            predictions={
                "prediction_1": results_83,
                "prediction_2": results_90
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/all")
async def predict_all(energy_inputs: List[EnergyInput], threshold_inputs: List[ThresholdInput]):
    """
    Combined prediction endpoint - both energy and threshold
    """
    try:
        # Get energy predictions
        energy_results = await predict_energy(energy_inputs)
        
        # Get threshold predictions  
        threshold_results = await predict_threshold(threshold_inputs)
        
        # Combine results
        return {
            "energy_predictions": energy_results,
            "threshold_predictions": threshold_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Energy Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health - Health check",
            "energy": "POST /predict/energy - Energy prediction", 
            "threshold": "POST /predict/threshold - Threshold prediction",
            "all": "POST /predict/all - Combined predictions",
            "docs": "GET /docs - Interactive API documentation"
        },
        "input_formats": {
            "energy": {
                "data": "2025-01-01",
                "boosting": 0.0,
                "cor": "incolor",
                "espessura": 8.0,
                "extracao_forno": 750.0,
                "porcentagem_caco": 15.0,
                "extracao_boosting": 1.5
            },
            "threshold": {
                "data": "2025-01-01",
                "cor": "incolor",
                "espessura": 8.0,
                "extracao_forno": 750.0,
                "porcentagem_caco": 15.0,
                "ext_boosting": 1.5,
                "prod_e": 1,
                "prod_l": 0,
                "autoclave": 1
            }
        }
    }

if __name__ == '__main__':
    import uvicorn
    
    print("🚀 Starting Simple Energy Prediction FastAPI...")
    print("📊 Available endpoints:")
    print("  GET  / - Root with API info")
    print("  GET  /health - Health check")
    print("  POST /predict/energy - Energy prediction") 
    print("  POST /predict/threshold - Threshold prediction")
    print("  POST /predict/all - Combined predictions")
    print("  GET  /docs - Interactive API documentation")
    print("\n🔧 Input format:")
    print('  Energy: [{"data": "2025-01-01", "boosting": 0.0, "cor": "incolor", "espessura": 8.0, "extracao_forno": 750.0, "porcentagem_caco": 15.0, "extracao_boosting": 1.5}]')
    print('  Threshold: [{"data": "2025-01-01", "cor": "incolor", "espessura": 8.0, "extracao_forno": 750.0, "porcentagem_caco": 15.0, "ext_boosting": 1.5, "prod_e": 1, "prod_l": 0, "autoclave": 1}]')
    print("\n✅ FastAPI ready on http://localhost:8000")
    print("📖 Interactive docs at http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)