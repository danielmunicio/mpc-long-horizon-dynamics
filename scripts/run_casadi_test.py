import sys
import os
import glob
import numpy as np
from mpc import MPC

def find_latest_experiment(resources_path):
    experiments = glob.glob(resources_path + "experiments/*/")
    if not experiments:
        raise FileNotFoundError(f"No experiments found in {resources_path}experiments/")
    return max(experiments, key=os.path.getctime)

if __name__ == '__main__':
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"

    exp = find_latest_experiment(resources_path)
    ckpt = max(glob.glob(exp + "checkpoints/*.pth"), key=os.path.getctime)

    start = np.zeros(14)

    controller = MPC(ckpt, exp, ckpt, exp, start)
    controller.test_casadi_function()
