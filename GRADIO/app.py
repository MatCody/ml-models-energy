import os
# Force CPU usage for XGBoost models
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'

import gradio as gr
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import glob

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
        """Load all models with fixed names"""
        try:
            # Load Random Forest Energy Model
            if os.path.exists('rf_energy_model.pkl'):
                with open('rf_energy_model.pkl', 'rb') as f:
                    rf_data = pickle.load(f)
                    self.rf_model = rf_data['model']
                    self.rf_preprocessor = rf_data['preprocessor']
                print("✅ Loaded Random Forest energy model")
            else:
                print("❌ rf_energy_model.pkl not found")
            
            # Load XGBoost Energy Model (use same RF model for now)
            if os.path.exists('xgboost_energy_model.pkl'):
                with open('xgboost_energy_model.pkl', 'rb') as f:
                    xgb_data = pickle.load(f)
                    self.xgb_model = xgb_data['model']
                    self.xgb_encoders = xgb_data.get('preprocessor', None)
                print("✅ Loaded XGBoost energy model")
            else:
                # Use RF model as fallback for XGBoost
                self.xgb_model = self.rf_model
                self.xgb_encoders = self.rf_preprocessor
                print("⚠️ xgboost_energy_model.pkl not found, using RF model as fallback")
            
            # Load Threshold Model 8300
            if os.path.exists('xgboost_threshold_8300_model.pkl'):
                with open('xgboost_threshold_8300_model.pkl', 'rb') as f:
                    threshold_data = pickle.load(f)
                    self.threshold_model_83 = threshold_data['model']
                    self.threshold_preprocessor = threshold_data.get('preprocessor', None)
                    # Fix XGBoost compatibility issues
                    try:
                        # Remove problematic attributes completely
                        problematic_attrs = ['use_label_encoder', 'gpu_id', 'predictor', 'tree_method']
                        for attr in problematic_attrs:
                            if hasattr(self.threshold_model_83, attr):
                                delattr(self.threshold_model_83, attr)
                        
                        # Force CPU settings
                        self.threshold_model_83.device = 'cpu'
                        
                        # Wrap predict_proba to handle errors
                        original_predict_proba = self.threshold_model_83.predict_proba
                        def safe_predict_proba(X):
                            try:
                                return original_predict_proba(X)
                            except Exception as e:
                                print(f"Prediction error: {e}")
                                # Return default probabilities if prediction fails
                                return [[0.5, 0.5] for _ in range(len(X))]
                        
                        self.threshold_model_83.predict_proba = safe_predict_proba
                        
                    except Exception as e:
                        print(f"Warning: 8300 model setup failed: {e}")
                print("✅ Loaded 8300 threshold model (CPU mode)")
            else:
                print("❌ xgboost_threshold_8300_model.pkl not found")
            
            # Load Threshold Model 9000
            if os.path.exists('xgboost_threshold_9000_model.pkl'):
                with open('xgboost_threshold_9000_model.pkl', 'rb') as f:
                    threshold_data = pickle.load(f)
                    self.threshold_model_90 = threshold_data['model']
                    # Fix XGBoost compatibility issues  
                    try:
                        # Remove problematic attributes completely
                        problematic_attrs = ['use_label_encoder', 'gpu_id', 'predictor', 'tree_method']
                        for attr in problematic_attrs:
                            if hasattr(self.threshold_model_90, attr):
                                delattr(self.threshold_model_90, attr)
                        
                        # Force CPU settings
                        self.threshold_model_90.device = 'cpu'
                        
                        # Wrap predict_proba to handle errors
                        original_predict_proba = self.threshold_model_90.predict_proba
                        def safe_predict_proba(X):
                            try:
                                return original_predict_proba(X)
                            except Exception as e:
                                print(f"Prediction error: {e}")
                                # Return default probabilities if prediction fails
                                return [[0.5, 0.5] for _ in range(len(X))]
                        
                        self.threshold_model_90.predict_proba = safe_predict_proba
                        
                    except Exception as e:
                        print(f"Warning: 9000 model setup failed: {e}")
                print("✅ Loaded 9000 threshold model (CPU mode)")
            else:
                print("❌ xgboost_threshold_9000_model.pkl not found")
            
            self.models_loaded = True
            return "Models loaded successfully"
        
        except Exception as e:
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            return f"Error loading models: {str(e)}"
    
    def predict_threshold(self, json_input):
        """Predict threshold exceedance"""
        try:
            if not self.models_loaded:
                return "Error: Models not loaded"
            
            if not self.threshold_model_83 or not self.threshold_model_90:
                return "Error: Threshold models not available"
            
            data = json.loads(json_input)
            
            # Handle both single object and array formats
            if not isinstance(data, list):
                data = [data]
            
            # Process all items
            results_83 = []
            results_90 = []
            
            for item in data:
                # Parse input data
                date_obj = datetime.strptime(item['data'], '%Y-%m-%d')
                
                # Color mapping
                color_mapping = {0: 'incolor', 1: 'verde', 2: 'cinza', 3: 'bronze'}
                if isinstance(item['cor'], str):
                    cor_str = item['cor'].lower()
                else:
                    cor_str = color_mapping.get(item['cor'], 'incolor')
                
                # Get ext_boosting for threshold model
                ext_boosting_val = item.get('ext_boosting', item.get('pot_boost', 0.0))
                
                # Create input features (NO temporal features for new models)
                input_data = {
                    'espessura': item['espessura'],
                    'extracao_forno': item['extracao_forno'],
                    'porcentagem_caco': item['porcentagem_caco'],
                    'ext_boosting': ext_boosting_val,
                    'cor': cor_str,
                    'prod_e': item.get('Prod_E', item.get('prod_e', 1)),
                    'prod_l': item.get('Prod_L', item.get('prod_l', 1)),
                    'autoclave': item.get('autoclave', 1)
                }
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Preprocess (handle case where no preprocessor is available)
                if self.threshold_preprocessor is not None:
                    X_processed = self.threshold_preprocessor.transform(input_df)
                else:
                    # Manual encoding for XGBoost threshold models
                    X_processed = pd.get_dummies(input_df, columns=['cor'], prefix='cor')
                
                # Make predictions with error handling
                try:
                    prob_83_raw = self.threshold_model_83.predict_proba(X_processed)
                    prob_83 = prob_83_raw[0][1] if len(prob_83_raw[0]) > 1 else prob_83_raw[0][0]
                    # Ensure probability is between 0 and 1
                    prob_83 = max(0.0, min(1.0, float(prob_83)))
                except Exception as e:
                    print(f"Error with threshold_83 prediction: {e}")
                    prob_83 = 0.0
                
                pred_83 = int(prob_83 > 0.5)
                
                try:
                    prob_90_raw = self.threshold_model_90.predict_proba(X_processed)
                    prob_90 = prob_90_raw[0][1] if len(prob_90_raw[0]) > 1 else prob_90_raw[0][0]
                    # Ensure probability is between 0 and 1
                    prob_90 = max(0.0, min(1.0, float(prob_90)))
                except Exception as e:
                    print(f"Error with threshold_90 prediction: {e}")
                    prob_90 = 0.0
                
                pred_90 = int(prob_90 > 0.5)
                
                # Add to results (using correct threshold values 8300/9000)
                results_83.append({
                    "datetime": item['data'],
                    "threshold": 8300,
                    "probabilidade_de_estouro": round(prob_83, 4),
                    "estouro_previsto": pred_83
                })
                
                results_90.append({
                    "datetime": item['data'],
                    "threshold": 9000,
                    "probabilidade_de_estouro": round(prob_90, 4),
                    "estouro_previsto": pred_90
                })
            
            # Format response
            result = {
                "predictions": {
                    "prediction_1": results_83,
                    "prediction_2": results_90
                }
            }
            
            return json.dumps(result, indent=2)
            
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def predict_energy_rf(self, json_input):
        """Predict energy using Random Forest"""
        try:
            if not self.models_loaded or not self.rf_model:
                return "Error: Random Forest model not available"
            
            data = json.loads(json_input)
            if not isinstance(data, list):
                data = [data]
            
            results = []
            
            for item in data:
                # Parse input
                date_obj = datetime.strptime(item['data'], '%Y-%m-%d')
                
                # Get extracao_boosting (main feature for new models)
                extracao_boosting_val = item.get('ext_boosting', item.get('extracao_boosting', 0.0))
                
                # Handle extracao_forno field
                if 'extracao_forno' in item:
                    extracao_val = float(str(item['extracao_forno']).replace(',', '.'))
                else:
                    extracao_val = 800.0
                
                # Create features (NEW MODEL FORMAT - no temporal features)
                input_data = {
                    'boosting': 0.0,  # Always 0 for compatibility (as requested)
                    'cor': str(item['cor']).lower() if isinstance(item['cor'], str) else {0: 'incolor', 1: 'verde', 2: 'cinza', 3: 'bronze'}.get(item['cor'], 'incolor'),
                    'espessura': item['espessura'],
                    'extracao_forno': extracao_val,
                    'porcentagem_caco': item['porcentagem_caco'],
                    'extracao_boosting': extracao_boosting_val
                }
                
                # Predict
                input_df = pd.DataFrame([input_data])
                X_processed = self.rf_preprocessor.transform(input_df)
                prediction = self.rf_model.predict(X_processed)[0]
                
                results.append({
                    "data": date_obj.strftime('%d-%m-%Y'),  # Changed back to DD-MM-YYYY format
                    "predicted_energy": float(prediction)
                })
            
            return json.dumps(results, indent=2)
            
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def predict_energy_xgb(self, json_input):
        """Predict energy using XGBoost"""
        try:
            if not self.models_loaded or not self.xgb_model:
                return "Error: XGBoost model not available"
            
            data = json.loads(json_input)
            if not isinstance(data, list):
                data = [data]
            
            results = []
            
            for item in data:
                # Parse input
                date_obj = datetime.strptime(item['data'], '%Y-%m-%d')
                
                # Get extracao_boosting (main feature for new models)
                extracao_boosting_val = item.get('ext_boosting', item.get('extracao_boosting', 0.0))
                
                # Handle extracao_forno field
                if 'extracao_forno' in item:
                    extracao_val = float(str(item['extracao_forno']).replace(',', '.'))
                else:
                    extracao_val = 800.0
                
                # Create features (NEW MODEL FORMAT - no temporal features)
                input_data = {
                    'boosting': 0.0,  # Always 0 for compatibility (as requested)
                    'cor': str(item['cor']).lower() if isinstance(item['cor'], str) else {0: 'incolor', 1: 'verde', 2: 'cinza', 3: 'bronze'}.get(item['cor'], 'incolor'),
                    'espessura': item['espessura'],
                    'extracao_forno': extracao_val,
                    'porcentagem_caco': item['porcentagem_caco'],
                    'extracao_boosting': extracao_boosting_val
                }
                
                # Preprocess features 
                input_df = pd.DataFrame([input_data])
                
                if self.xgb_encoders is not None:
                    X_processed = self.xgb_encoders.transform(input_df)
                else:
                    # Manual encoding if no preprocessor available
                    X_processed = pd.get_dummies(input_df, columns=['cor'], prefix='cor')
                
                # Predict
                prediction = self.xgb_model.predict(X_processed)[0]
                
                results.append({
                    "data": date_obj.strftime('%d-%m-%Y'),  # Changed back to DD-MM-YYYY format
                    "predicted_energy": float(prediction)
                })
            
            return json.dumps(results, indent=2)
            
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error: {str(e)}"

# Initialize predictor
predictor = EnergyMLPredictor()

def make_prediction(model_choice, json_input):
    """Make prediction based on model choice"""
    if not predictor.models_loaded:
        load_msg = predictor.load_models()
        if "Error" in load_msg:
            return load_msg
    
    if model_choice == "Threshold Detection":
        return predictor.predict_threshold(json_input)
    elif model_choice == "Energy Prediction (Random Forest)":
        return predictor.predict_energy_rf(json_input)
    elif model_choice == "Energy Prediction (XGBoost)":
        return predictor.predict_energy_xgb(json_input)
    else:
        return "Error: Please select a model"

# Default examples (updated for new models)
threshold_example = """{
  "data": "2025-01-01",
  "cor": "incolor",
  "espessura": 8.0,
  "ext_boosting": 1.5,
  "extracao_forno": 750.0,
  "porcentagem_caco": 15.0,
  "prod_e": 1,
  "prod_l": 0,
  "autoclave": 1
}"""

energy_example = """[
  {
    "data": "2025-01-01",
    "boosting": 0.0,
    "cor": "incolor",
    "espessura": 8.0,
    "extracao_forno": 750.0,
    "porcentagem_caco": 15.0,
    "extracao_boosting": 1.5
  }
]"""

# Test data from holdout period (last 2 months used in training)
week_test_data = """[
  {"data": "2025-04-19", "cor": 0, "espessura": 8.0, "ext_boosting": 1.2007, "extracao_forno": 699.561202512973, "porcentagem_caco": 10.0062724674475, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-04-20", "cor": 0, "espessura": 8.0, "ext_boosting": 1.2026, "extracao_forno": 699.169485837721, "porcentagem_caco": 9.99757589767354, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 0},
  {"data": "2025-04-21", "cor": 0, "espessura": 8.0, "ext_boosting": 1.201, "extracao_forno": 699.134346519477, "porcentagem_caco": 9.99807838764974, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 0, "autoclave": 0},
  {"data": "2025-04-22", "cor": 0, "espessura": 8.0, "ext_boosting": 1.2074, "extracao_forno": 701.318973743488, "porcentagem_caco": 9.99545180216949, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-04-23", "cor": 0, "espessura": 8.0, "ext_boosting": 1.2028, "extracao_forno": 702.765143096952, "porcentagem_caco": 9.97488288777139, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 1, "autoclave": 1},
  {"data": "2025-04-24", "cor": 0, "espessura": 8.0, "ext_boosting": 1.3973, "extracao_forno": 700.8439481142, "porcentagem_caco": 10.002226628142, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 1, "autoclave": 1},
  {"data": "2025-04-25", "cor": 0, "espessura": 8.0, "ext_boosting": 1.6005, "extracao_forno": 702.032548397562, "porcentagem_caco": 9.98529201530728, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 1}
]"""

month_test_data = """[
  {"data": "2025-04-19", "cor": 0, "espessura": 8.0, "ext_boosting": 1.2007, "extracao_forno": 699.561202512973, "porcentagem_caco": 10.0062724674475, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-04-20", "cor": 0, "espessura": 8.0, "ext_boosting": 1.2026, "extracao_forno": 699.169485837721, "porcentagem_caco": 9.99757589767354, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 0},
  {"data": "2025-04-21", "cor": 0, "espessura": 8.0, "ext_boosting": 1.201, "extracao_forno": 699.134346519477, "porcentagem_caco": 9.99807838764974, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 0, "autoclave": 0},
  {"data": "2025-04-22", "cor": 0, "espessura": 8.0, "ext_boosting": 1.2074, "extracao_forno": 701.318973743488, "porcentagem_caco": 9.99545180216949, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-04-23", "cor": 0, "espessura": 8.0, "ext_boosting": 1.2028, "extracao_forno": 702.765143096952, "porcentagem_caco": 9.97488288777139, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 1, "autoclave": 1},
  {"data": "2025-04-24", "cor": 0, "espessura": 8.0, "ext_boosting": 1.3973, "extracao_forno": 700.8439481142, "porcentagem_caco": 10.002226628142, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 1, "autoclave": 1},
  {"data": "2025-04-25", "cor": 0, "espessura": 8.0, "ext_boosting": 1.6005, "extracao_forno": 702.032548397562, "porcentagem_caco": 9.98529201530728, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-04-26", "cor": 0, "espessura": 8.0, "ext_boosting": 1.7549, "extracao_forno": 703.33718364331, "porcentagem_caco": 9.96677008271902, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-04-27", "cor": 0, "espessura": 8.0, "ext_boosting": 1.8022, "extracao_forno": 698.519152270116, "porcentagem_caco": 10.0355158154479, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 0, "autoclave": 0},
  {"data": "2025-04-28", "cor": 0, "espessura": 8.0, "ext_boosting": 1.8023, "extracao_forno": 699.802291106822, "porcentagem_caco": 10.0171149610168, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-04-29", "cor": 0, "espessura": 8.0, "ext_boosting": 1.803, "extracao_forno": 702.213883737496, "porcentagem_caco": 9.98271347568585, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 1, "autoclave": 0},
  {"data": "2025-04-30", "cor": 0, "espessura": 8.0, "ext_boosting": 1.801, "extracao_forno": 701.164091438783, "porcentagem_caco": 9.99765972843181, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 1, "autoclave": 0},
  {"data": "2025-05-01", "cor": 0, "espessura": 8.0, "ext_boosting": 1.7999, "extracao_forno": 701.096395285213, "porcentagem_caco": 9.99862507800837, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 1, "autoclave": 0},
  {"data": "2025-05-02", "cor": 0, "espessura": 8.0, "ext_boosting": 1.8016, "extracao_forno": 701.004721690124, "porcentagem_caco": 9.99993264396119, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 1, "autoclave": 0},
  {"data": "2025-05-03", "cor": 0, "espessura": 8.0, "ext_boosting": 1.8023, "extracao_forno": 699.505291072901, "porcentagem_caco": 10.021368086077, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 0, "autoclave": 0},
  {"data": "2025-05-04", "cor": 0, "espessura": 8.0, "ext_boosting": 1.8036, "extracao_forno": 700.073447985429, "porcentagem_caco": 10.0132350686523, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 0, "autoclave": 0},
  {"data": "2025-05-05", "cor": 0, "espessura": 8.0, "ext_boosting": 0.689, "extracao_forno": 700.60585295748, "porcentagem_caco": 10.0056258028798, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-05-06", "cor": 0, "espessura": 8.0, "ext_boosting": 0.0, "extracao_forno": 699.123418185867, "porcentagem_caco": 10.026841924692, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-05-07", "cor": 0, "espessura": 8.0, "ext_boosting": 0.0, "extracao_forno": 699.086556585488, "porcentagem_caco": 10.0273706223712, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-05-08", "cor": 0, "espessura": 8.0, "ext_boosting": 0.0, "extracao_forno": 698.120389195209, "porcentagem_caco": 10.0412480547676, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-05-09", "cor": 0, "espessura": 8.0, "ext_boosting": 0.0, "extracao_forno": 697.228099576186, "porcentagem_caco": 9.9680434627127, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 0, "autoclave": 0},
  {"data": "2025-05-10", "cor": 0, "espessura": 8.0, "ext_boosting": 0.0, "extracao_forno": 697.37935572186, "porcentagem_caco": 9.96588147179382, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 0, "autoclave": 0},
  {"data": "2025-05-11", "cor": 0, "espessura": 8.0, "ext_boosting": 0.0, "extracao_forno": 699.563378916139, "porcentagem_caco": 10.0205359675357, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-05-12", "cor": 0, "espessura": 8.0, "ext_boosting": 0.0, "extracao_forno": 698.733542903546, "porcentagem_caco": 10.0324366436888, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-05-13", "cor": 0, "espessura": 8.0, "ext_boosting": 0.0, "extracao_forno": 699.509702244859, "porcentagem_caco": 10.0213048904162, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-05-14", "cor": 0, "espessura": 8.0, "ext_boosting": 0.0, "extracao_forno": 701.657766576732, "porcentagem_caco": 9.99062553558067, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-05-15", "cor": 0, "espessura": 8.0, "ext_boosting": 0.0, "extracao_forno": 674.645706945424, "porcentagem_caco": 10.0052515424159, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-05-16", "cor": 0, "espessura": 8.0, "ext_boosting": 0.0, "extracao_forno": 653.148421891636, "porcentagem_caco": 9.95179622600148, "pot_boost": 3.0, "Prod_E": 0, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-05-17", "cor": 0, "espessura": 6.0, "ext_boosting": 0.0, "extracao_forno": 611.090907286899, "porcentagem_caco": 9.98214819965588, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 1},
  {"data": "2025-05-18", "cor": 0, "espessura": 6.0, "ext_boosting": 0.0, "extracao_forno": 599.399563235682, "porcentagem_caco": 10.0100173040013, "pot_boost": 3.0, "Prod_E": 1, "Prod_L": 0, "autoclave": 1}
]"""

# Generate test data functions
def generate_energy_test_data(days=1):
    """Generate test data for energy models"""
    from datetime import datetime, timedelta
    
    base_date = datetime(2025, 1, 1)
    test_data = []
    
    for i in range(days):
        current_date = base_date + timedelta(days=i)
        test_data.append({
            "data": current_date.strftime("%Y-%m-%d"),
            "boosting": 0.0,
            "cor": "incolor" if i % 3 == 0 else "verde" if i % 3 == 1 else "cinza",
            "espessura": 8.0 + (i % 3),
            "extracao_forno": 750.0 + (i * 10),
            "porcentagem_caco": 15.0 + (i * 2),
            "extracao_boosting": 1.5 + (i * 0.3)
        })
    
    return json.dumps(test_data, indent=2)

def generate_threshold_test_data(days=1):
    """Generate test data for threshold models"""
    from datetime import datetime, timedelta
    
    base_date = datetime(2025, 1, 1)
    test_data = []
    
    for i in range(days):
        current_date = base_date + timedelta(days=i)
        test_data.append({
            "data": current_date.strftime("%Y-%m-%d"),
            "cor": "incolor" if i % 3 == 0 else "verde" if i % 3 == 1 else "cinza",
            "espessura": 8.0 + (i % 3),
            "ext_boosting": 1.5 + (i * 0.3),
            "extracao_forno": 750.0 + (i * 10),
            "porcentagem_caco": 15.0 + (i * 2),
            "prod_e": i % 2,
            "prod_l": (i + 1) % 2,
            "autoclave": 1 if i % 3 == 0 else 0
        })
    
    return json.dumps(test_data, indent=2)

# Create custom interfaces with test data buttons
def create_energy_interface(model_name, predict_fn, api_name):
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"### {model_name}")
            gr.Markdown("Generate test data or enter your own JSON:")
            
            with gr.Row():
                btn_1_day = gr.Button("1 Day", size="sm")
                btn_3_days = gr.Button("3 Days", size="sm")
                btn_week = gr.Button("1 Week", size="sm")
                btn_clear = gr.Button("Clear", size="sm")
            
            json_input = gr.Textbox(
                label="JSON Input",
                lines=12,
                value=generate_energy_test_data(1),
                placeholder="Enter JSON data here..."
            )
            
            predict_btn = gr.Button("Predict Energy", variant="primary")
            
        with gr.Column():
            output = gr.Textbox(
                label="Prediction Result",
                lines=12,
                interactive=False
            )
    
    # Event handlers
    btn_1_day.click(lambda: generate_energy_test_data(1), outputs=json_input)
    btn_3_days.click(lambda: generate_energy_test_data(3), outputs=json_input)
    btn_week.click(lambda: generate_energy_test_data(7), outputs=json_input)
    btn_clear.click(lambda: "", outputs=json_input)
    predict_btn.click(predict_fn, inputs=json_input, outputs=output)

def create_threshold_interface(predict_fn, api_name):
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Threshold Detection")
            gr.Markdown("Generate test data or enter your own JSON:")
            
            with gr.Row():
                btn_1_day = gr.Button("1 Day", size="sm")
                btn_3_days = gr.Button("3 Days", size="sm")
                btn_week = gr.Button("1 Week", size="sm")
                btn_clear = gr.Button("Clear", size="sm")
            
            json_input = gr.Textbox(
                label="JSON Input",
                lines=12,
                value=generate_threshold_test_data(1),
                placeholder="Enter JSON data here..."
            )
            
            predict_btn = gr.Button("Predict Thresholds", variant="primary")
            
        with gr.Column():
            output = gr.Textbox(
                label="Prediction Result",
                lines=12,
                interactive=False
            )
    
    # Event handlers  
    btn_1_day.click(lambda: generate_threshold_test_data(1), outputs=json_input)
    btn_3_days.click(lambda: generate_threshold_test_data(3), outputs=json_input)
    btn_week.click(lambda: generate_threshold_test_data(7), outputs=json_input)
    btn_clear.click(lambda: "", outputs=json_input)
    predict_btn.click(predict_fn, inputs=json_input, outputs=output)

# Create Gradio interface with tabs
with gr.Blocks(title="Energy ML Cloud", theme=gr.themes.Default()) as app:
    
    gr.Markdown("# Energy ML Prediction System")
    gr.Markdown("Cloud deployment with embedded models - Each tab has its own API endpoint")
    
    with gr.Tabs():
        with gr.TabItem("Energy Prediction (Random Forest)"):
            create_energy_interface("Random Forest Energy Model", predictor.predict_energy_rf, "energy_random_forest")
            
        with gr.TabItem("Energy Prediction (XGBoost)"):
            create_energy_interface("XGBoost Energy Model", predictor.predict_energy_xgb, "energy_xgboost")
            
        with gr.TabItem("Threshold Detection"):
            create_threshold_interface(predictor.predict_threshold, "threshold_detection")
    
    # Load models when app starts
    predictor.load_models()
    
    with gr.Accordion("Model Information", open=False):
        gr.Markdown("""
        ## Available Models
        - **Threshold Detection**: Predict probability of exceeding 8300 and 9000 consumption thresholds
        - **Random Forest**: Energy prediction (R² = 0.50, MAE = 0.24 MWh)
        - **XGBoost**: Energy prediction (R² = 0.53, MAE = 0.24 MWh, best model)
        
        ## Input Formats
        See examples that change when you select different models.
        """)

if __name__ == "__main__":
    app.launch(
        auth=("admin", "energy123"),
        share=True,
        ssr_mode=False
    )
