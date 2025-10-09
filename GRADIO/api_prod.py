from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import pickle
import pandas as pd
from datetime import datetime
import os

app = FastAPI(title="Energy ML API Vivix")

class EnergyMLPredictor:
    def __init__(self):
        self.rf_model = None
        self.rf_preprocessor = None
        self.xgb_model = None
        self.xgb_encoders = None
        self.threshold_model_83 = None
        self.threshold_model_90 = None
        self.threshold_preprocessor = None
        self.models_loaded = False
        
    def load_models(self):
        try:
            if os.path.exists('rf_energy_model.pkl'):
                with open('rf_energy_model.pkl', 'rb') as f:
                    rf_data = pickle.load(f)
                    self.rf_model = rf_data['model']
                    self.rf_preprocessor = rf_data['preprocessor']
            
            if os.path.exists('xgboost_energy_model.pkl'):
                with open('xgboost_energy_model.pkl', 'rb') as f:
                    xgb_data = pickle.load(f)
                    self.xgb_model = xgb_data['model']
                    self.xgb_encoders = xgb_data['label_encoders']
            
            if os.path.exists('threshold_model_83_autoclave.pkl'):
                with open('threshold_model_83_autoclave.pkl', 'rb') as f:
                    threshold_data = pickle.load(f)
                    self.threshold_model_83 = threshold_data['model']
                    self.threshold_preprocessor = threshold_data['preprocessor']
            
            if os.path.exists('threshold_model_90_autoclave.pkl'):
                with open('threshold_model_90_autoclave.pkl', 'rb') as f:
                    threshold_data = pickle.load(f)
                    self.threshold_model_90 = threshold_data['model']
            
            self.models_loaded = True
            return {"status": "success", "message": "Models loaded successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def predict_threshold(self, data):
        if not self.models_loaded or not self.threshold_model_83 or not self.threshold_model_90:
            return {"error": "Threshold models not available"}
        
        if not isinstance(data, list):
            data = [data]
        
        results_83, results_90 = [], []
        for item in data:
            date_obj = datetime.strptime(item['data'], '%Y-%m-%d')
            color_mapping = {0: 'incolor', 1: 'verde', 2: 'cinza', 3: 'bronze'}
            cor_str = item['cor'].lower() if isinstance(item['cor'], str) else color_mapping.get(item['cor'], 'incolor')
            boosting_val = item.get('pot_boost', item.get('ext_boosting', 3.0))

            input_data = {
                'boosting': boosting_val,
                'espessura': item['espessura'],
                'extracao_forno': item['extracao_forno'],
                'porcentagem_caco': item['porcentagem_caco'],
                'cor': cor_str,
                'prod_e': item.get('Prod_E', item.get('prod_e', 1)),
                'prod_l': item.get('Prod_L', item.get('prod_l', 1)),
                'autoclave': item.get('autoclave', 1),
                'week_day': date_obj.weekday(),
                'month': date_obj.month,
                'quarter': (date_obj.month - 1) // 3 + 1,
                'is_weekend': int(date_obj.weekday() >= 5),
                'week_of_year': date_obj.isocalendar()[1]
            }

            input_df = pd.DataFrame([input_data])
            X_processed = self.threshold_preprocessor.transform(input_df)

            try:
                prob_83 = float(self.threshold_model_83.predict_proba(X_processed)[0][1])
            except Exception:
                prob_83 = 0.0
            pred_83 = int(prob_83 > 0.5)

            try:
                prob_90 = float(self.threshold_model_90.predict_proba(X_processed)[0][1])
            except Exception:
                prob_90 = 0.0
            pred_90 = int(prob_90 > 0.5)

            results_83.append({"datetime": item['data'], "probabilidade_de_estouro": round(prob_83, 4), "estouro_previsto": pred_83})
            results_90.append({"datetime": item['data'], "probabilidade_de_estouro": round(prob_90, 4), "estouro_previsto": pred_90})
        
        return {"predictions": {"prediction_1": results_83, "prediction_2": results_90}}

    def predict_energy_rf(self, data):
        if not self.models_loaded or not self.rf_model:
            return {"error": "Random Forest model not available"}
        
        if not isinstance(data, list):
            data = [data]

        results = []
        for item in data:
            date_obj = datetime.strptime(item['data'], '%Y-%m-%d')
            boosting_val = float(str(item.get('boosting', item.get('ext_boosting', 0))).replace(',', '.'))
            extracao_val = float(str(item.get('extracao_forno', 800.0)).replace(',', '.'))

            input_data = {
                'boosting': boosting_val,
                'espessura': item['espessura'],
                'extracao_forno': extracao_val,
                'porcentagem_caco': item['porcentagem_caco'],
                'cor': str(item['cor']).lower() if isinstance(item['cor'], str) else {0: 'incolor',1:'verde',2:'cinza',3:'bronze'}.get(item['cor'], 'incolor'),
                'prod_e': item.get('prod_e', item.get('Prod_E', 1)),
                'prod_l': item.get('prod_l', item.get('Prod_L', 1)),
                'autoclave': item.get('autoclave', 1),
                'week_day': date_obj.weekday(),
                'month': date_obj.month,
                'quarter': (date_obj.month - 1)//3+1,
                'is_weekend': int(date_obj.weekday()>=5),
                'week_of_year': date_obj.isocalendar()[1],
                'day_of_month': date_obj.day,
                'day_of_year': date_obj.timetuple().tm_yday
            }
            input_df = pd.DataFrame([input_data])
            X_processed = self.rf_preprocessor.transform(input_df)
            prediction = self.rf_model.predict(X_processed)[0]
            results.append({"data": date_obj.strftime('%d-%m-%Y'), "predictions": float(prediction)})
        return results

    def predict_energy_xgb(self, data):
        if not self.models_loaded or not self.xgb_model:
            return {"error": "XGBoost model not available"}
        
        if not isinstance(data, list):
            data = [data]

        results = []
        for item in data:
            date_obj = datetime.strptime(item['data'], '%Y-%m-%d')
            boosting_val = float(str(item.get('boosting', item.get('ext_boosting', 0))).replace(',', '.'))
            extracao_val = float(str(item.get('extracao_forno', 800.0)).replace(',', '.'))

            input_data = {
                'boosting': boosting_val,
                'espessura': item['espessura'],
                'extracao_forno': extracao_val,
                'porcentagem_caco': item['porcentagem_caco'],
                'cor': str(item['cor']).lower() if isinstance(item['cor'], str) else {0: 'incolor',1:'verde',2:'cinza',3:'bronze'}.get(item['cor'], 'incolor'),
                'week_day': date_obj.weekday(),
                'month': date_obj.month,
                'quarter': (date_obj.month - 1)//3+1,
                'week_of_year': date_obj.isocalendar()[1],
                'prod_e': item.get('prod_e', 1),
                'prod_l': item.get('prod_l', 1),
                'is_weekend': int(date_obj.weekday()>=5),
                'autoclave': item.get('autoclave', 1)
            }
            input_df = pd.DataFrame([input_data])
            for col in input_df.columns:
                if col in self.xgb_encoders:
                    try:
                        input_df[col] = self.xgb_encoders[col].transform(input_df[col].astype(str))
                    except ValueError:
                        input_df[col] = 0
            prediction = self.xgb_model.predict(input_df.values)[0]
            results.append({"data": date_obj.strftime('%d-%m-%Y'), "predictions": float(prediction)})
        return results

predictor = EnergyMLPredictor()

@app.post("/load-models")
def load_models():
    result = predictor.load_models()
    return JSONResponse(content=result)


# -----------------------------
# Example week_test_data payload, remember the zero boosting...
# -----------------------------
week_test_example = [
    {
        "data": "2023-01-01",
        "cor": "incolor",
        "espessura": 8.0,
        "ext_boosting": 65.0,
        "extracao_forno": 851.1,
        "porcentagem_caco": 15.0,
        "pot_boost": 0.0,
        "Prod_E": 1,
        "Prod_L": 1,
        "autoclave": 1
    },
    {
        "data": "2023-01-02",
        "cor": "verde",
        "espessura": 10.0,
        "ext_boosting": 60.0,
        "extracao_forno": 820.0,
        "porcentagem_caco": 10.0,
        "pot_boost": 0.0,
        "Prod_E": 1,
        "Prod_L": 1,
        "autoclave": 1
    }
]


@app.post("/predict/threshold")
def predict_threshold(payload: dict = Body(..., example=week_test_example)):
    result = predictor.predict_threshold(payload)
    return JSONResponse(content=result)

@app.post("/predict/energy-rf")
def predict_energy_rf(payload: dict = Body(..., example=week_test_example)):
    result = predictor.predict_energy_rf(payload)
    return JSONResponse(content=result)

@app.post("/predict/energy-xgb")
def predict_energy_xgb(payload: dict = Body(..., example=week_test_example)):
    result = predictor.predict_energy_xgb(payload)
    return JSONResponse(content=result)

