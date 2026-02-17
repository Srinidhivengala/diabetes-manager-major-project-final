"""
Federated Model Service
Wrapper to use federated learning model in the Flask application
"""
import os
import joblib
import numpy as np
from typing import Dict, Tuple, List
from sklearn.linear_model import LogisticRegression

DEFAULT_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]


class FederatedModelService:
    """Service for using federated learning model"""
    
    def __init__(self, model_path: str = None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.model_path = model_path or os.path.join(base_dir, 'ml', 'federated_model.pkl')
        self.model = self._load_model()
    
    def _load_model(self):
        """Load federated model or return fallback"""
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                print(f"Loaded federated model from {self.model_path}")
                return model
            except Exception as e:
                print(f"Error loading federated model: {e}")
        
        # Fallback to simple rule-based model
        print("Using fallback model (federated model not found)")
        class SimpleGlucoseRule:
            def predict_proba(self, X):
                glucose_idx = DEFAULT_FEATURES.index('Glucose')
                probs = []
                for row in X:
                    g = row[glucose_idx]
                    p = 0.2 if g < 120 else 0.7 if g < 155 else 0.9
                    probs.append([1 - p, p])
                return np.array(probs)
        return SimpleGlucoseRule()
    
    def predict(self, payload: Dict) -> Tuple[int, float, List[str]]:
        """Make prediction using federated model"""
        features = [payload.get(k, 0) for k in DEFAULT_FEATURES]
        X = np.array(features, dtype=float).reshape(1, -1)
        
        try:
            proba = self.model.predict_proba(X)[0][1]
        except:
            # Fallback if model doesn't have predict_proba
            proba = self.model.predict(X)[0]
            proba = float(proba) if isinstance(proba, (int, float)) else 0.5
        
        pred = int(proba >= 0.5)
        return pred, float(proba), DEFAULT_FEATURES
    
    def is_available(self) -> bool:
        """Check if federated model is available"""
        return os.path.exists(self.model_path)

