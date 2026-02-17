"""
Federated Learning Client using Flower
Each client (hospital/clinic) trains on their local data without sharing it
"""
import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict, List, Tuple, Optional
import os

FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]
TARGET = 'Outcome'


class FederatedClient(fl.client.NumPyClient):
    """Federated Learning Client"""
    
    def __init__(self, data_path: str, client_id: str = "client_1"):
        """
        Initialize client with local data
        
        Args:
            data_path: Path to local CSV data file
            client_id: Unique identifier for this client
        """
        self.client_id = client_id
        self.data_path = data_path
        self.X_train, self.y_train, self.X_test, self.y_test = self.load_data()
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        
        print(f"Client {client_id} initialized with {len(self.X_train)} training samples")
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare local data"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        df = df.dropna()
        
        # Ensure all required features exist
        missing_features = [f for f in FEATURES if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        X = df[FEATURES].values.astype(np.float32)
        y = df[TARGET].values.astype(np.int32) if TARGET in df.columns else np.zeros(len(X), dtype=np.int32)
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_test, y_test
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model parameters"""
        if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
            return [self.model.coef_[0].astype(np.float32), self.model.intercept_.astype(np.float32)]
        # Return initial parameters if model not trained yet
        return [np.zeros(len(FEATURES), dtype=np.float32), np.array([0.0], dtype=np.float32)]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from server"""
        weights, bias = parameters
        self.model.coef_ = weights.reshape(1, -1)
        self.model.intercept_ = bias
        self.model.classes_ = np.array([0, 1])
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on local data"""
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Get training config
        local_epochs = config.get("local_epochs", 5)
        
        # Train model
        for epoch in range(local_epochs):
            self.model.fit(self.X_train, self.y_train)
        
        # Return updated parameters and metrics
        parameters_updated = self.get_parameters(config)
        num_samples = len(self.X_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(self.X_train)
        train_acc = accuracy_score(self.y_train, train_pred)
        
        metrics = {
            "train_accuracy": float(train_acc),
            "client_id": self.client_id,
            "samples": num_samples
        }
        
        print(f"Client {self.client_id}: Trained on {num_samples} samples, Train Acc: {train_acc:.3f}")
        
        return parameters_updated, num_samples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model on local test data"""
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Evaluate
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Calculate AUC if possible
        try:
            auc = roc_auc_score(self.y_test, y_pred_proba)
        except:
            auc = 0.0
        
        num_samples = len(self.X_test)
        
        metrics = {
            "test_accuracy": float(accuracy),
            "test_auc": float(auc),
            "client_id": self.client_id
        }
        
        print(f"Client {self.client_id}: Test Acc: {accuracy:.3f}, Test AUC: {auc:.3f}")
        
        # Return loss (1 - accuracy), number of samples, and metrics
        loss = 1.0 - accuracy
        return float(loss), num_samples, metrics


def main():
    """Run a federated learning client"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--data", type=str, required=True, help="Path to local data CSV file")
    parser.add_argument("--server", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--client-id", type=str, default="client_1", help="Client identifier")
    args = parser.parse_args()
    
    # Create client
    client = FederatedClient(args.data, args.client_id)
    
    # Connect to server
    print(f"Connecting to server at {args.server}...")
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client
    )
    print(f"Client {args.client_id} finished")


if __name__ == "__main__":
    main()

