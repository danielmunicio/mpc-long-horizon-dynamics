import casadi as ca
import torch
import l4casadi as l4c
import sys 
import os 
import glob
from config import load_args
from dynamics_learning.lighting import DynamicsLearning

class MPC():
    def __init__(self, checkpoint_path, experiment_path):
        self.model, self.args = self.load_model(checkpoint_path, experiment_path)
        l4c_model = l4c.L4CasADi(self.model, device='cpu')
        self.opti = ca.Opti()

        self.v = ca.SX.sym('v', 3)      # linear velocity
        self.q = ca.SX.sym('q', 4)      # quaternion
        self.w = ca.SX.sym('w', 3)      # angular velocity
        self.u = ca.SX.sym('u', 4)      # control input

        # Flattened history window: (history_length * 14,) for l4casadi compatibility
        self.x = ca.SX.sym('x', self.args.history_length * 14)

        # Model takes history and outputs next velocity (or attitude)
        self.f = l4c_model(self.x)
        self.f_func = ca.Function('learned_dynamics', [self.x], [self.f])
        self.test_casadi_function()

    def load_model(self, checkpoint_path, experiment_path):
        args = load_args(os.path.join(experiment_path, 'args.txt'))

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

    def test_casadi_function(self):
        # Test with dummy flattened history: (history_length * 14,)
        x_test = ca.DM.zeros(self.args.history_length * 14)
        print("Testing CasADi function...")
        result = self.f_func(x_test)
        print("Output shape:", result.shape)
        print("Output:", result)


if __name__ == '__main__':
    # Find latest experiment
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    experiment_path = max(glob.glob(resources_path + "experiments/*/"), key=os.path.getctime)
    checkpoint_path = max(glob.glob(experiment_path + "checkpoints/*.pth"), key=os.path.getctime)

    print(f"Loading model from: {checkpoint_path}")
    controller = MPC(checkpoint_path, experiment_path)


