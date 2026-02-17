import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

DEFAULT_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]


class ModelService:
    def __init__(self, model_path: str = None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.model_path = model_path or os.path.join(base_dir, 'ml', 'model.pkl')
        self.model = self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        # Fallback simple model: threshold on glucose using median-like rule
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
        features = [payload.get(k, 0) for k in DEFAULT_FEATURES]
        X = np.array(features, dtype=float).reshape(1, -1)
        proba = self.model.predict_proba(X)[0][1]
        pred = int(proba >= 0.5)
        return pred, float(proba), DEFAULT_FEATURES
