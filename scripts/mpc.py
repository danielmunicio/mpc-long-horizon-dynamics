import casadi as ca
import l4casadi as l4c
import numpya s np
import sys 
import os 
import glob
from config import load_args
from dynamics_learning.lightning import DynamicsLearning

class MPC():
    def __init__(self, checkpoint_path, experiment_path):
         self.model, self.args = self.load_model(checkpoint_path, experiment_path)
        l4c_model = l4c.L4CasADi(model, device='cpu')
       self.opti = ca.Opti()
        self.p = ca.SX.sym('p', 3)
        self.v = ca.SX.sym('v', 3)
        self.q = ca.SX.sym('q', 4)
        self.w = ca.SX.sym('w', 3)

        self.x = ca.vertcat(p, v, q, w)
        
        self.u = ca.SX.sym('u', 4)


        # I think this is how you do this ? 
        self.f = l4c_model(self.x)
        self.f_func = ca.Function('learned_dynamics', [self.x, self.u], self.f)

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


