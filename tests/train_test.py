# Much around with the path to make the import work
import os
import sys
from pathlib import Path

import omegaconf
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf.dictconfig import DictConfig

# Add the parent directory to the path to import the script. Hacky, but it works.
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent))

from scripts.train import main


def _discover_training_configs():
    parent_dirs = [
        "commands/dedo/hangproccloth_unimodal",
        "commands/dedo/hangproccloth_simple",
        "commands/dedo/hangproccloth_multimodal",
        "commands/dedo/hangbag",
    ]

    config_names = []
    for parent_dir in parent_dirs:
        # Find all files recursively in the parent directory.
        for root, _, files in os.walk(f"configs/{parent_dir}"):
            for file in files:
                if file.endswith("train.yaml"):
                    config_names.append(os.path.join(root[8:], file[:-5]))

    return config_names


GOLDEN_TRAIN_CFGS = _discover_training_configs()


@pytest.mark.parametrize("config_name", GOLDEN_TRAIN_CFGS)
def test_train_cfgs(config_name):
    with initialize(config_path=str("../configs")):
        cfg: DictConfig = compose(
            config_name=config_name,
            overrides=[
                # Ugly things we need to add for some unknown reason.
                "hydra.verbose=true",
                "hydra.job.num=1",
                "hydra.job.id=0",
                "hydra.hydra_help.hydra_help=nil",  # Hack to avoid hydra help.
                "hydra.runtime.output_dir=.",
                "seed=1234",
                "training.batch_size=2",
                "dataset.data_dir=/path/to/data",
            ],
            return_hydra_config=True,
        )
        # Resolve the config
        HydraConfig.instance().set_config(cfg)

        # Resolve the config!
        container = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )


# Skip if CLOTH_DATASET_PATH is not set
@pytest.mark.skipif(
    not os.environ.get("CLOTH_DATASET_PATH"), reason="CLOTH_DATASET_PATH not set"
)
@pytest.mark.parametrize("config_name", GOLDEN_TRAIN_CFGS)
def test_train(config_name):
    dataset_dir = os.environ.get("CLOTH_DATASET_PATH")

    with initialize(config_path=str("../configs")):
        cfg: DictConfig = compose(
            config_name=config_name,
            overrides=[
                "hydra.verbose=true",
                "hydra.job.num=1",
                "hydra.runtime.output_dir=.",
                "seed=1234",
                "training.batch_size=2",
                f"dataset.data_dir={dataset_dir}",
            ],
            return_hydra_config=True,
        )
        # Resolve the config
        HydraConfig.instance().set_config(cfg)

        os.environ["WANDB_MODE"] = "disabled"
        main(cfg)
