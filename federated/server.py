"""
Federated Learning Server using Flower
This server coordinates federated training across multiple clients (hospitals/clinics)
"""
import flwr as fl
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Model parameters
FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]
NUM_FEATURES = len(FEATURES)


def get_initial_parameters():
    """Initialize model parameters"""
    # For logistic regression: weights + bias
    # Weights shape: (num_features,), Bias: scalar
    weights = np.zeros(NUM_FEATURES, dtype=np.float32)
    bias = np.array([0.0], dtype=np.float32)
    return [weights, bias]


def set_model_params(model, params):
    """Set model parameters from federated learning results"""
    weights, bias = params
    model.coef_ = weights.reshape(1, -1)
    model.intercept_ = bias


def get_model_params(model):
    """Extract model parameters for federated learning"""
    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        return [model.coef_[0].astype(np.float32), model.intercept_.astype(np.float32)]
    return get_initial_parameters()


class FederatedServer:
    """Federated Learning Server"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join('ml', 'federated_model.pkl')
        self.strategy = FedAvg(
            fraction_fit=0.3,  # Use 30% of available clients per round
            fraction_evaluate=0.3,  # Evaluate on 30% of clients
            min_fit_clients=2,  # Minimum 2 clients needed
            min_evaluate_clients=2,
            min_available_clients=2,  # Wait for at least 2 clients
            initial_parameters=fl.common.ndarrays_to_parameters(get_initial_parameters()),
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
        )
    
    def fit_config(self, server_round: int):
        """Return training configuration for each round"""
        return {
            "server_round": server_round,
            "local_epochs": 5,  # Number of local training epochs
        }
    
    def evaluate_config(self, server_round: int):
        """Return evaluation configuration"""
        return {"server_round": server_round}
    
    def save_model(self, parameters):
        """Save the aggregated model"""
        # Convert Flower parameters to model
        weights, bias = parameters
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.coef_ = weights.reshape(1, -1)
        model.intercept_ = bias
        model.classes_ = np.array([0, 1])
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        print(f"Saved federated model to {self.model_path}")


def main():
    """Run the federated learning server"""
    server = FederatedServer()
    
    # Custom strategy wrapper to save model after each round
    class ModelSavingStrategy(fl.server.strategy.Strategy):
        def __init__(self, base_strategy, save_callback):
            self.base_strategy = base_strategy
            self.save_callback = save_callback
        
        def initialize_parameters(self, client_manager):
            return self.base_strategy.initialize_parameters(client_manager)
        
        def configure_fit(self, server_round, parameters, client_manager):
            return self.base_strategy.configure_fit(server_round, parameters, client_manager)
        
        def aggregate_fit(self, server_round, results, failures):
            aggregated = self.base_strategy.aggregate_fit(server_round, results, failures)
            # Save model after aggregation
            if aggregated and aggregated[0]:
                try:
                    parameters = fl.common.parameters_to_ndarrays(aggregated[0])
                    self.save_callback(parameters, server_round)
                except Exception as e:
                    print(f"Error saving model: {e}")
            return aggregated
        
        def configure_evaluate(self, server_round, parameters, client_manager):
            return self.base_strategy.configure_evaluate(server_round, parameters, client_manager)
        
        def aggregate_evaluate(self, server_round, results, failures):
            return self.base_strategy.aggregate_evaluate(server_round, results, failures)
        
        def evaluate(self, server_round, parameters):
            return self.base_strategy.evaluate(server_round, parameters)
    
    # Wrap strategy with model saving
    def save_model_callback(parameters, round_num):
        server.save_model(parameters)
        print(f"Round {round_num}: Model saved successfully")
    
    saving_strategy = ModelSavingStrategy(server.strategy, save_model_callback)
    
    # Start server
    print("Starting Federated Learning Server...")
    print("Waiting for clients to connect...")
    print("Server will run for 10 rounds")
    print("Server address: 0.0.0.0:8080")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=saving_strategy,
    )


if __name__ == "__main__":
    main()

