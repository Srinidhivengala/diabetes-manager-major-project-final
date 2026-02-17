# Federated Learning Implementation

This directory contains the federated learning implementation for the Diabetes Management System.

## Overview

Federated Learning allows multiple hospitals/clinics to collaboratively train a machine learning model without sharing their sensitive patient data. Each client trains on local data, and only model parameters (not data) are shared with the central server.

## Architecture

- **Server** (`server.py`): Coordinates federated training, aggregates model updates
- **Client** (`client.py`): Trains model on local data, sends updates to server
- **Simulation** (`../federated_sim.py`): Simulates multiple clients for testing

## How It Works

1. **Server** initializes a global model
2. **Clients** download the global model
3. **Clients** train the model on their local data
4. **Clients** send model updates (parameters) to server
5. **Server** aggregates updates using FedAvg (Federated Averaging)
6. **Server** distributes updated model to clients
7. Repeat steps 2-6 for multiple rounds

## Usage

### 1. Start the Server

```bash
python federated/server.py
```

The server will listen on `0.0.0.0:8080` and wait for clients to connect.

### 2. Start Clients

In separate terminals, start each client:

```bash
# Client 1
python federated/client.py --data data/pima_client_1.csv --client-id client_1 --server localhost:8080

# Client 2
python federated/client.py --data data/pima_client_2.csv --client-id client_2 --server localhost:8080

# Client 3
python federated/client.py --data data/pima_client_3.csv --client-id client_3 --server localhost:8080
```

### 3. Run Simulation (Easier)

For testing, use the simulation script:

```bash
python federated_sim.py --rounds 10 --clients 3
```

This will:
- Automatically split data into client datasets
- Start the server
- Start multiple clients
- Run federated training for specified rounds

## Model Output

After training, the federated model is saved to:
- `ml/federated_model.pkl`

The Flask application will automatically use this model if available, falling back to the regular model if not.

## Privacy Benefits

- **No Data Sharing**: Raw patient data never leaves each hospital/clinic
- **Only Parameters Shared**: Only model weights/parameters are transmitted
- **Differential Privacy**: Can be extended with differential privacy for additional protection
- **Secure Aggregation**: Model updates can be encrypted before transmission

## Configuration

Edit `server.py` to adjust:
- `num_rounds`: Number of federated learning rounds
- `fraction_fit`: Fraction of clients used per round
- `min_fit_clients`: Minimum clients required
- `local_epochs`: Training epochs per client per round

## Requirements

- Flower (flwr): `pip install flwr`
- scikit-learn
- numpy
- pandas

## Integration with Flask App

The Flask app automatically detects and uses the federated model:
- If `ml/federated_model.pkl` exists, it's used for predictions
- Otherwise, falls back to the regular RandomForest model
- No code changes needed in the Flask app

