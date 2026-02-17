from typing import Dict, List
import numpy as np

try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def explain_prediction(model, payload: Dict, features_order: List[str]):
    x = np.array([[float(payload.get(k, 0)) for k in features_order]])
    if not SHAP_AVAILABLE:
        # Simple heuristic explanation proportional to feature values
        vals = x[0] / (np.linalg.norm(x[0]) + 1e-8)
        return {f: float(v) for f, v in zip(features_order, vals)}

    try:
        explainer = shap.Explainer(model)
        sv = explainer(x)
        shap_values = sv.values[0] if hasattr(sv, 'values') else sv[0].values
        return {f: float(v) for f, v in zip(features_order, shap_values)}
    except Exception:
        vals = x[0] / (np.linalg.norm(x[0]) + 1e-8)
        return {f: float(v) for f, v in zip(features_order, vals)}
