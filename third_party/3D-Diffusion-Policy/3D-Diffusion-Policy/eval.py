if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
# from train import TrainDP3Workspace, EvalTAX3DWorkspace
from retreiveGoalPC import TrainDP3Workspace, EvalTAX3DWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):

    if cfg.name == "tax3d":
        workspace = EvalTAX3DWorkspace(cfg)
    else:
        workspace = TrainDP3Workspace(cfg)
    
    # workspace.eval()
    workspace.eval_datasets()

if __name__ == "__main__":
    main()
