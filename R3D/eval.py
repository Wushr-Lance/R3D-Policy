if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainDP3Workspace

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'r3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.eval()

class DP3_policy:
    def __init__(self, cfg, checkpoint_num, run_dir) -> None:
        self.cfg = cfg
        self.checkpoint_num = checkpoint_num
        self.run_dir = run_dir
        self.n_obs_steps = cfg.n_obs_steps

    def get_policy(self):
        eval_py_dir = str(pathlib.Path(__file__).parent)
        run_dir = os.path.join('data/outputs', self.run_dir)
        workspace = TrainDP3Workspace(self.cfg, os.path.join(eval_py_dir, run_dir))
        policy = workspace.get_policy(self.cfg, self.checkpoint_num) 
        n_obs_steps = self.n_obs_steps
        return policy, n_obs_steps


if __name__ == "__main__":
    main()
