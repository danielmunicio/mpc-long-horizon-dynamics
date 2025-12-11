import casadi as ca
import torch
import l4casadi as l4c
import sys 
import os 
import glob
from config import load_args
from dynamics_learning.lighting import DynamicsLearning

class MPC():
    def __init__(self, velocity_checkpoint, velocity_experiment, attitude_checkpoint, attitude_experiment, x0):
        self.velocity_model, self.args = self.load_model(velocity_checkpoint, velocity_experiment)
        self.attitude_model, _ = self.load_model(attitude_checkpoint, attitude_experiment)

        l4c_velocity = l4c.L4CasADi(self.velocity_model, device='cpu')
        l4c_attitude = l4c.L4CasADi(self.attitude_model, device='cpu')

        self.opti = ca.Opti()

        # Flattened history window: (history_length * 14,) for l4casadi compatibility
        self.x = ca.SX.sym('x', self.args.history_length * 14)

        # Model functions
        self.f_velocity = l4c_velocity(self.x)
        self.f_attitude = l4c_attitude(self.x)
        self.f_velocity_func = ca.Function('velocity_dynamics', [self.x], [self.f_velocity])
        self.f_attitude_func = ca.Function('attitude_dynamics', [self.x], [self.f_attitude])

        self.history = []
        self.initialize_history(x0)

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

    def initialize_history(self, x0):
        """Fill history buffer with initial state (or oldest state if history exists)"""
        if len(self.history) == 0:
            fill_state = x0
        else:
            fill_state = self.history[0]

        while len(self.history) < self.args.history_length:
            self.history.append(fill_state.copy() if hasattr(fill_state, 'copy') else fill_state)

    def add_to_history(self, state):
        """Add new state to history, maintaining sliding window"""
        self.history.append(state)
        if len(self.history) > self.args.history_length:
            self.history.pop(0)

    def get_flattened_history(self):
        """Return flattened history for l4casadi model"""
        return ca.vertcat(*self.history)

    def test_casadi_function(self):
        x_test = ca.DM.zeros(self.args.history_length * 14)
        print("Testing CasADi function...")
        result = self.f_func(x_test)
        print("Output shape:", result.shape)
        print("Output:", result)

    def build_optimization_problem(self, goal, N=10):
        X = []
        U = []

        Xf = self.opti.parameter(14)
        self.opti.set_value(Xf, goal)

        history_buffer = []
        for i in range(self.args.history_length):
            h = self.opti.parameter(14)
            self.opti.set_value(h, self.history[i])
            history_buffer.append(h)

        # Build decision variables and dynamics constraints
        cost = 0
        for k in range(N):
            X_k = self.opti.variable(14)
            U_k = self.opti.variable(4)
            X.append(X_k)
            U.append(U_k)

            # Flatten history window for model
            history_flat = ca.vertcat(*history_buffer)

            # Predict velocity and attitude
            velocity_pred = self.f_velocity_func(history_flat)
            attitude_pred = self.f_attitude_func(history_flat)

            # velocity_pred is [v_next(3), w_next(3)]
            # attitude_pred is [q_next(4)]
            v_next = velocity_pred[:3]
            w_next = velocity_pred[3:6]
            q_next = attitude_pred[:4]

            X_next = ca.vertcat(v_next, q_next, w_next, U_k)

            # Dynamics constraint
            self.opti.subject_to(X_k == X_next)

            # Input constraints
            self.opti.subject_to(U_k >= ca.DM(self.u_lower_bounds))
            self.opti.subject_to(U_k <= ca.DM(self.u_upper_bounds))

            # State constraints
            self.opti.subject_to(X_k >= ca.DM(self.x_lower_bounds))
            self.opti.subject_to(X_k <= ca.DM(self.x_upper_bounds))

            # Cost function
            cost += ca.sumsqr(X_k - Xf) + 0.01 * ca.sumsqr(U_k)

            # Slide history window forward
            history_buffer = history_buffer[1:] + [X_k]

        self.opti.minimize(cost)
        self.opti.solver("ipopt")
        sol = self.opti.solve()

        return sol.value(ca.hcat(X)), sol.value(ca.hcat(U))

if __name__ == '__main__':
    # Find latest experiment
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    experiment_path = max(glob.glob(resources_path + "experiments/*/"), key=os.path.getctime)
    checkpoint_path = max(glob.glob(experiment_path + "checkpoints/*.pth"), key=os.path.getctime)

    start = np.array([])
    goal = np.array([])


    controller = MPC(checkpoint_path, experiment_path)
    controller.build_optimization_problem(start, goal)

