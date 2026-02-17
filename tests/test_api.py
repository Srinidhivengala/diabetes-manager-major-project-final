import json
import os
import pytest

from app import app, db, seed_defaults


@pytest.fixture(scope='module')
def test_client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.app_context():
        db.create_all()
        seed_defaults()
    with app.test_client() as client:
        yield client


def login(client, email, password):
    return client.post('/login', data={'email': email, 'password': password}, follow_redirects=True)


def test_health(test_client):
    rv = test_client.get('/health')
    assert rv.status_code == 200


def test_predict_flow(test_client):
    login(test_client, 'patient@example.com', 'password123')
    payload = {
        'Pregnancies': 1, 'Glucose': 130, 'BloodPressure': 70, 'SkinThickness': 20,
        'Insulin': 80, 'BMI': 28, 'DiabetesPedigreeFunction': 0.5, 'Age': 35
    }
    rv = test_client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
    assert rv.status_code == 200
    data = rv.get_json()
    assert 'prediction' in data and 'probability' in data

    rv2 = test_client.post('/api/explain', data=json.dumps(payload), content_type='application/json')
    assert rv2.status_code == 200
    data2 = rv2.get_json()
    assert 'shap_values' in data2

    rv3 = test_client.post('/api/forecast', data=json.dumps({'readings': []}), content_type='application/json')
    assert rv3.status_code == 200
    data3 = rv3.get_json()
    assert 'forecast' in data3
