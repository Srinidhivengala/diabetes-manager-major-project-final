from typing import List, Dict
from datetime import datetime, timedelta
import numpy as np


def _parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.now()


def forecast_glucose(readings: List[Dict]) -> List[Dict]:
    # Expect list of {timestamp, glucose}
    if not readings:
        now = datetime.now()
        return [{
            'timestamp': (now + timedelta(minutes=5*i)).isoformat(),
            'glucose': 110.0
        } for i in range(1, 13)]

    # sort by time
    readings = sorted(readings, key=lambda r: _parse_ts(r['timestamp']))
    y = np.array([float(r['glucose']) for r in readings], dtype=float)

    # moving average smoothing
    window = min(5, len(y))
    if window > 1:
        kernel = np.ones(window) / window
        y_smooth = np.convolve(y, kernel, mode='same')
    else:
        y_smooth = y

    # linear regression on index -> glucose
    x = np.arange(len(y_smooth)).reshape(-1, 1)
    x = np.hstack([np.ones_like(x), x])
    coef, _, _, _ = np.linalg.lstsq(x, y_smooth, rcond=None)

    last_ts = _parse_ts(readings[-1]['timestamp'])
    forecasts: List[Dict] = []
    last_idx = len(y_smooth) - 1
    for i in range(1, 13):  # next 60 minutes in 5-min steps
        idx = last_idx + i
        pred = float(coef[0] + coef[1] * idx)
        ts = last_ts + timedelta(minutes=5 * i)
        forecasts.append({'timestamp': ts.isoformat(), 'glucose': max(60.0, min(pred, 300.0))})
    return forecasts
