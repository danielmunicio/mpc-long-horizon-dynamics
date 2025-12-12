# scripts/mpc.py
import casadi as ca
import torch
import l4casadi as l4c
import numpy as np
import sys
import os
import glob

from config import load_args
from dynamics_learning.lighting import DynamicsLearning


# -----------------------------
# Quaternion utilities (xyzw)
# -----------------------------
def quat_mul(q1, q2):
    """
    Hamilton product for quaternions in (x, y, z, w) ordering.
    q = [qx, qy, qz, qw]^T
    """
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
    return ca.vertcat(
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_normalize(q):
    # Safe normalization for CasADi derivatives
    norm_sq = ca.sumsqr(q)
    return ca.if_else(
        norm_sq > 1e-6,
        q / ca.sqrt(norm_sq),
        ca.vertcat(0, 0, 0, 1)
    )


# -----------------------------
# Checkpoint inspection helpers
# -----------------------------
def infer_output_size_from_checkpoint(state_dict):
    """
    Infer output dimension from final Linear weight in state_dict.
    Works with your DynamicsLearning models where the output head is a Linear layer.
    """
    linear_weights = []
    for k, v in state_dict.items():
        if k.endswith("weight") and hasattr(v, "ndim") and v.ndim == 2:
            linear_weights.append((k, tuple(v.shape)))

    if not linear_weights:
        raise RuntimeError("Could not infer output size: no 2D weight tensors found in state_dict")

    last_name, last_shape = linear_weights[-1]
    out_dim = last_shape[0]
    return out_dim, last_name


def load_latest_checkpoint(exp_path):
    ckpts = glob.glob(os.path.join(exp_path, "checkpoints", "*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {exp_path}/checkpoints/")
    return max(ckpts, key=os.path.getctime)


def find_latest_experiment_by_type(experiments_root, predictor_type):
    exp_paths = glob.glob(os.path.join(experiments_root, "*", ""))
    if not exp_paths:
        raise FileNotFoundError(f"No experiments found under {experiments_root}")

    candidates = []
    for ep in exp_paths:
        args_path = os.path.join(ep, "args.txt")
        if not os.path.exists(args_path):
            continue
        try:
            a = load_args(args_path)
            if getattr(a, "predictor_type", None) == predictor_type:
                candidates.append(ep)
        except Exception:
            pass

    if not candidates:
        raise FileNotFoundError(f"No experiments found with predictor_type={predictor_type}")

    exp = max(candidates, key=os.path.getctime)
    ckpt = load_latest_checkpoint(exp)
    return exp, ckpt


# =============================
# MPC
# =============================
class MPC:
    """
    State record per timestep (14D):
        [ v(3), q(4), w(3), u_prev(4) ]

    Decision control per step:
        u_k (4)

    Learned model I/O (from your checkpoints):
        velocity model: (6,)  -> [dv(3), dw(3)]
        attitude model: (4,)  -> dq(4) used as q_{k+1} = dq ⊙ q_k
    """

    def __init__(self, vel_ckpt, vel_exp, att_ckpt, att_exp, x0_14, H_eff=4):
        # Load models + infer output dims from checkpoints (robust)
        self.vel_model, self.vel_args, self.vel_out_dim = self.load_model(vel_ckpt, vel_exp)
        self.att_model, self.att_args, self.att_out_dim = self.load_model(att_ckpt, att_exp)

        if self.vel_out_dim != 6:
            raise ValueError(f"Velocity model output dim expected 6, got {self.vel_out_dim}")
        if self.att_out_dim != 4:
            raise ValueError(f"Attitude model output dim expected 4, got {self.att_out_dim}")

        # Use velocity args as canonical (both should have same history length)
        self.args = self.vel_args
        self.H = int(self.args.history_length)
        self.H_eff = int(min(max(1, H_eff), self.H))

        # Precompute padding selector matrix (Option 2)
        # xhist_padded = S @ xhist_full
        nx = 14
        H = self.H
        He = self.H_eff
        S = np.zeros((H * nx, H * nx), dtype=float)

        # padding rows: repeat first state x0
        for i in range(H - He):
            S[i * nx:(i + 1) * nx, 0:nx] = np.eye(nx)

        # recent rows: keep last He states
        for i in range(He):
            src = (H - He + i) * nx
            dst = (H - He + i) * nx
            S[dst:dst + nx, src:src + nx] = np.eye(nx)

        self.pad_selector = ca.DM(S)

        # Wrap with L4CasADi (IMPORTANT: distinct names to avoid cache collisions)
        self.vel_l4c = l4c.L4CasADi(self.vel_model, device="cpu", name="velocity_model")
        self.att_l4c = l4c.L4CasADi(self.att_model, device="cpu", name="attitude_model")

        # Build CasADi functions once (MX graph)
        xhist_full = ca.MX.sym("xhist_full", self.H * 14)
        xhist = self.pad_selector @ xhist_full

        vel_out = ca.reshape(self.vel_l4c(xhist), self.vel_out_dim, 1)
        att_out = ca.reshape(self.att_l4c(xhist), self.att_out_dim, 1)

        self.f_vel = ca.Function("f_vel", [xhist_full], [vel_out])
        self.f_att = ca.Function("f_att", [xhist_full], [att_out])

        # History buffer (numpy)
        self.history = []
        self.initialize_history(np.array(x0_14, dtype=float))

        # Bounds
        self.u_lb = np.array([-5, -5, 0, -5]).reshape(4, 1)
        self.u_ub = np.array([5, 5, 20, 5]).reshape(4, 1)

    def load_model(self, ckpt_path, exp_path):
        args = load_args(os.path.join(exp_path, "args.txt"))

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]

        out_dim, _ = infer_output_size_from_checkpoint(state_dict)

        folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
        resources_path = folder_path + "resources/"

        model = DynamicsLearning(
            args,
            resources_path,
            exp_path,
            input_size=14,
            output_size=out_dim,
            max_iterations=1,
        )

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        return model, args, out_dim

    # -----------------------------
    # History handling
    # -----------------------------
    def initialize_history(self, x0):
        self.history = []
        for _ in range(self.H):
            self.history.append(np.array(x0, dtype=float))

    def add_to_history(self, x):
        self.history.append(np.array(x, dtype=float))
        if len(self.history) > self.H:
            self.history.pop(0)

    def _history_params(self, opti):
        hist_p = [opti.parameter(14, 1) for _ in range(self.H)]
        for i in range(self.H):
            opti.set_value(hist_p[i], self.history[i].reshape(14, 1))
        return hist_p

    # -----------------------------
    # MPC solve (N=2, soft dynamics)
    # -----------------------------
    def solve(self, goal_14, N=2, w_dyn=1e4):
        """
        Soft dynamics MPC.
        goal_14: numpy (14,)
        returns:
            X_opt: (14, N+1)
            U_opt: (4, N)
        """
        N = int(N)
        if N <= 0:
            raise ValueError("N must be >= 1")
        goal_14 = np.array(goal_14, dtype=float).reshape(14,)

        opti = ca.Opti()

        # Parameters: measured history and goal
        hist_p = self._history_params(opti)

        goal_p = opti.parameter(14, 1)
        opti.set_value(goal_p, goal_14.reshape(14, 1))

        # Decision variables
        X = [opti.variable(14, 1) for _ in range(N + 1)]
        U = [opti.variable(4, 1) for _ in range(N)]

        # Initial state equals last measured record
        opti.subject_to(X[0] == hist_p[-1])

        rolling = list(hist_p)

        # Cost
        w_u = 1e-2
        cost = 0

        for k in range(N):
            xk = X[k]
            uk = U[k]

            v_k = xk[0:3]
            q_k = xk[3:7]
            w_k = xk[7:10]

            # Record uses decision control
            rec_k = ca.vertcat(v_k, q_k, w_k, uk)

            # Full history window (H records)
            window = rolling[-(self.H - 1):] + [rec_k]
            xhist_full = ca.vertcat(*window)  # (H*14, 1)

            # Learned predictions (models see padded history via selector inside f_vel/f_att)
            vel_pred = self.f_vel(xhist_full)  # (6,1)
            att_pred = self.f_att(xhist_full)  # (4,1)

            dv = vel_pred[0:3]
            dw = vel_pred[3:6]

            v_next = v_k + dv
            w_next = w_k + dw

            dq = att_pred
            dq = ca.fmin(ca.fmax(dq, -2.0), 2.0)
            q_next = quat_normalize(quat_mul(dq, q_k))

            x_next = ca.vertcat(v_next, q_next, w_next, uk)

            # hard dynamics constraint
            opti.subject_to(X[k+1] == x_next)

            # Control bounds
            opti.subject_to(uk >= self.u_lb)
            opti.subject_to(uk <= self.u_ub)

            # Tracking + effort
            cost += ca.sumsqr(X[k + 1] - goal_p) + w_u * ca.sumsqr(uk)

            

            # Roll forward
            rolling.append(X[k + 1])

        # Terminal cost
        cost += 10.0 * ca.sumsqr(X[-1] - goal_p)

        opti.minimize(cost)

        opti.solver(
            "ipopt",
            {"expand": True},
            {
                "print_level": 0,
                "max_iter": 100,
                "tol": 1e-3,
                "acceptable_tol": 1e-1,
                "acceptable_iter": 5,
                "hessian_approximation": "limited-memory",
            },
        )

        # Warm-start
        x_init = self.history[-1].reshape(14, 1)
        for k in range(N + 1):
            opti.set_initial(X[k], x_init)
        for k in range(N):
            opti.set_initial(U[k], np.zeros((4, 1)))

        sol = opti.solve()

        X_opt = np.array(sol.value(ca.hcat(X)))  # (14, N+1)
        U_opt = np.array(sol.value(ca.hcat(U)))  # (4, N)
        return X_opt, U_opt

    def rollout_one_step(self, xk, uk):
        """
        Roll learned dynamics forward one step (uses the same padded-history model call).
        """
        v_k = xk[0:3]
        q_k = xk[3:7]
        w_k = xk[7:10]

        rec_k = np.concatenate([v_k, q_k, w_k, uk])
        hist = self.history[-(self.H - 1):] + [rec_k]
        xhist_full = ca.DM(np.concatenate(hist))

        vel_pred = self.f_vel(xhist_full).full().flatten()
        att_pred = self.f_att(xhist_full).full().flatten()

        dv = vel_pred[0:3]
        dw = vel_pred[3:6]

        v_next = v_k + dv
        w_next = w_k + dw

        dq = ca.DM(att_pred)
        dq = ca.fmin(ca.fmax(dq, -2.0), 2.0)
        q_next = quat_normalize(quat_mul(dq, ca.DM(q_k))).full().flatten()

        x_next = np.concatenate([v_next, q_next, w_next, uk])
        return x_next


# =============================
# Entry point
# =============================
if __name__ == "__main__":
    print("Finding experiments...")

    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    experiments_root = os.path.join(folder_path, "resources", "experiments")

    vel_exp, vel_ckpt = find_latest_experiment_by_type(experiments_root, "velocity")
    att_exp, att_ckpt = find_latest_experiment_by_type(experiments_root, "attitude")

    print("Velocity experiment:", vel_exp)
    print("Velocity checkpoint:", vel_ckpt)
    print("Attitude experiment:", att_exp)
    print("Attitude checkpoint:", att_ckpt)

    # Initial record [v,q,w,u_prev]
    x0 = np.zeros(14)
    x0[3:7] = np.array([0, 0, 0, 1], dtype=float)

    # Goal record (example)
    goal = np.zeros(14)
    goal[3:7] = np.array([0, 0, 0, 1], dtype=float)

    # Reduce effective history (padding), but keep trained history length for the model
    controller = MPC(vel_ckpt, vel_exp, att_ckpt, att_exp, x0, H_eff=4)
    print("Initialized MPC controller")

    print("\n===== 10-step MPC rollout (N=2, soft dynamics) =====")

    x = x0.copy()
    for t in range(10):
        X, U = controller.solve(goal, N=2, w_dyn=1e3)
        u0 = U[:, 0]

        x_next = controller.rollout_one_step(x, u0)

        dist = np.linalg.norm(x_next - goal)
        q_norm = np.linalg.norm(x_next[3:7])
        u_norm = np.linalg.norm(u0)

        print(
            f"step {t:02d} | "
            f"||x-goal|| = {dist:.4f} | "
            f"||q|| = {q_norm:.4f} | "
            f"||u|| = {u_norm:.4f}"
        )

        controller.add_to_history(x_next)
        x = x_next
