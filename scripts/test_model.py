import torch
import numpy as np
import sys
import os
import glob
from dynamics_learning.lighting import DynamicsLearning
from config import load_args

def load_model(checkpoint_path, experiment_path):
    """Load trained model from checkpoint"""

    # Load args from experiment
    args = load_args(os.path.join(experiment_path, "args.txt"))

    # Set paths
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"

    # Determine output size based on predictor type
    if args.predictor_type == "velocity":
        output_size = 6
    elif args.predictor_type == "attitude":
        output_size = 4
    else:
        raise ValueError(f"Unknown predictor type: {args.predictor_type}")

    # Initialize model
    model = DynamicsLearning(
        args,
        resources_path,
        experiment_path,
        input_size=14,
        output_size=output_size,
        max_iterations=1,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model, args

def predict_next_state(model, states):
    """
    Predict next state given history of states

    Args:
        model: Trained DynamicsLearning model
        states: torch.Tensor of shape [batch_size, history_length, 14]
                or [history_length, 14] for single prediction
                Format: [lin_vel(3), quat(4), ang_vel(3), controls(4)]

    Returns:
        prediction: Predicted next state component
                   - velocity predictor: [lin_vel(3), ang_vel(3)]
                   - attitude predictor: [quat(4)]
    """

    # Handle single batch
    if states.dim() == 2:
        states = states.unsqueeze(0)

    with torch.no_grad():
        prediction = model.forward(states, init_memory=True)

    return prediction


if __name__ == "__main__":
    # Example usage

    # Find latest experiment
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    experiment_path = max(glob.glob(resources_path + "experiments/*/"), key=os.path.getctime)
    checkpoint_path = max(glob.glob(experiment_path + "checkpoints/*.pth"), key=os.path.getctime)

    print(f"Loading model from: {checkpoint_path}")

    # Load model
    model, args = load_model(checkpoint_path, experiment_path)
    print(f"Model type: {args.model_type}")
    print(f"Predictor type: {args.predictor_type}")
    print(f"History length: {args.history_length}")

    # Create dummy input (batch_size=2, history_length=20, features=14)
    batch_size = 2
    history_length = args.history_length
    dummy_states = torch.randn(batch_size, history_length, 14)

    # Single-step prediction
    print("\n=== Single-step prediction ===")
    prediction = predict_next_state(model, dummy_states)
    print(f"Input shape: {dummy_states.shape}")
    print(f"Output shape: {prediction.shape}")
    print(f"Prediction:\n{prediction}")
