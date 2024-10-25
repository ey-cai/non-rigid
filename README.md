# TAX3D: Non-rigid Relative Placement through 3D Dense Diffusion #
Eric Cai, Octavian Donca, Ben Eisner, David Held


# Installation #

This repository can be conveniently installed as a Python package, and used in downstream projects.

## Create conda environment and install dependencies ##

For now, all of the up-to-date TAX3D code is in the ``articulated`` branch.
```
git clone https://github.com/ey-cai/non-rigid.git
cd non-rigid
git checkout articulated
conda create --name ENVNAME python=3.9 pip==23.3.1 setuptools==65.5
```
Before installing ``non-rigid``, you'll need to install versions of PyTorch, PyTorch Geometric, and PyTorch3D. We've provided specific GPU versions in ``requirements-gpu.txt``:
```
pip install -r requirements-gpu.txt
```

## Install DEDO and 3D Diffusion Policy ##

```
cd third_party/dedo
pip install -e .
cd ../..

cd third_party/3D-Diffusion-Policy/3D-Diffusion-Policy
pip install -e .
cd ../../..
```
Note: the TAX3D repo contains significant changes to the underlying code in both the DEDO and 3D Diffusion Policy repository (refer to the READMEs in the respective ``third_party/`` directories, will update in the future).


## Install ``non-rigid`` ##

For now, the easiest thing to do is to install ``non-rigid`` in editable mode. This should install all of the additional required dependencies, as listed in ``pyproject.toml``.

```
pip install -e .
```
And you're done!

# Generating Datasets #

For convenience, the exact datasets used to run all experiments in the paper can be found [here](https://www.dropbox.com/scl/fi/l0iz7fjs0ooewdbhv6x4w/tax3d_data.zip?rlkey=ccckswgltmsztbq99urw07k1t&st=c3oq7hyz&dl=0).

As a reference, these are the commands to re-generate the datasets:
```
# For HangProcCloth-simple
python third_party/3D-Diffusion-Policy/third_party/dedo_scripts/gen_demonstration_proccloth.py --root_dir="<path/to/data/directory>/proccloth" --num_episodes=[NUM_EPISODES_PER_SPLIT] --split=[SPLIT] --random_anchor_pose --cloth_hole=single

# For HangProcCloth-unimodal
python third_party/3D-Diffusion-Policy/third_party/dedo_scripts/gen_demonstration_proccloth.py --root_dir="<path/to/data/directory>/proccloth" --num_episodes=[NUM_EPISODES_PER_SPLIT] --split=[SPLIT] --random_anchor_pose --random_cloth_geometry --cloth_hole=single

# For HangProcCloth-multimodal
python third_party/3D-Diffusion-Policy/third_party/dedo_scripts/gen_demonstration_proccloth.py --root_dir="<path/to/data/directory>/proccloth" --num_episodes=[NUM_EPISODES_PER_SPLIT] --split=[SPLIT] --random_anchor_pose --random_cloth_geometry --cloth_hole=double

# For HangBag
python third_party/3D-Diffusion-Policy/third_party/dedo_scripts/gen_demonstration_hangbag.py --root_dir="<path/to/data/directory>/hangbag" --num_episodes=[NUM_EPISODES_PER_SPLIT] --split=[SPLIT] --random_anchor_pose --cloth_hole=single
```
As noted in the paper, `HangProcCloth-simple` and `HangBag` used `train/val/val_ood` split sizes of 16/40/40, while `HangProcCloth-unimodal` and `HangProcCloth-multimodal` used 64/40/40.

# Training Models #
To train a model, run:
```
./train.sh [GPU_INDEX] [MODEL_TYPE] [WANDB_MODE] [ADDITIONAL_OVERRIDES]
```
For example, to train a TAX3D-CD model with our exact training parameters and log results to WandB, run:
```
./train.sh 0 cross_flow_relative online dataset.train_size=400
```
In general, the following `MODEL_TYPE`s correspond to the following ablations/models in the paper:

Ablation Name | `MODEL_TYPE` 
-- | --
**Scene Displacement/Point (SD/SP)** | `scene_flow`/`scene_point`
**Cross Displacement/Point - World Frame (CD-W/CP-W)** | `cross_flow_absolute`/`cross_point_absolute`
**Regression Displacement/Point (RD/RP)** | `regression_flow`/`regression_point`
**TAX3D-CD/TAX3D-CP** | `cross_flow_relative`/`cross_point_relative`

To run the **Cross Displacement/Point - No Action Context (CD-NAC/CP-NAC)** ablations, you will need to additionally disable the action context encoder. This can be done by overriding the model config from the command line:
```
# For CD-NAC
./train.sh 0 cross_flow_relative online dataset.train_size=400 model.x0_encoder=null
# For CP-NAC
./train.sh 0 cross_point_relative online dataset.train_size=400 model.x0_encoder=null
```
Note: you may have to update the ``data_dir`` parameter in ``configs/dataset/proc_cloth.yaml`` to properly reflect the directory where your data is stored. This can also be done from the command line:
```
./train.sh 0 cross_flow_relative online dataet.train_size=400 dataset.data_dir=[PATH_TO_YOUR_DATASET]
```
The exact config structure (and what exactly can be overrided in `[ADDITIONAL_OVERRIDES]`) can be seen in the config structure in the `configs/` directory.

## Reproducing Experiments ##
```
# re-train TAX3D-CD model for Table 1: HangProcCloth-unimodal (swap out MODEL_TYPE appropriately)
./train.sh 0 cross_flow_relative online dataset.train_size=400 dataset.data_dir=[PATH_TO_PROCCLOTH_DATASETS] dataset.cloth_geometry=multi dataset.hole=single

# re-train TAX3D-CD model for Table 2: HangProcCloth-simple (swap out MODEL_TYPE appropriately)
./train.sh 0 cross_flow_relative online dataset.train_size=400 dataset.data_dir=[PATH_TO_PROCCLOTH_DATASETS] dataset.cloth_geometry=single dataset.hole=single

# re-train TAX3D-CD model for Table 3: HangProcCloth-multimodal (swap out MODEL_TYPE appropriately)
./train.sh 0 cross_flow_relative online dataset.train_size=400 dataset.data_dir=[PATH_TO_PROCCLOTH_DATASETS] dataset.cloth_geometry=multi dataset.hole=double

# re-train TAX3D-CD model for Table 4: HangBag (swap out MODEL_TYPE appropriately)
./train.sh 0 cross_flow_relative online dataset.train_size=400 dataset.data_dir=[PATH_TO_HANGBAG_DATASETS] dataset.cloth_geometry=single dataset.hole=single
```



# Running Evaluations #

## Point Prediction Evaluations ##
To get point prediction errors, run:
```
./eval.sh [GPU_INDEX] [WANDB_CHECKPOINT_RUN_ID] [METRIC]=True
```
For example, to get point predictions results for a TAX3D-CD checkpoint with run id `gzc40qe1`, run:
```
# for coverage
./eval.sh 0 gzc40qe1 coverage=True
# for precision
./eval.sh 0 gzc40qe1 precision=True
```
Note: the evaluation script parses dataset and model configs from the original training run, and evaluates on the original dataset/model architecture by default.

You may have to update the ``data_dir`` parameter in ``configs/dataset/proc_cloth.yaml`` to properly reflect the directory where your data is stored. This can also be done from the command line:
```
./multi_cloth_eval.sh 0 cross_flow_relative sfr4r4hs coverage=True dataset.data_dir=[PATH_TO_YOUR_DATASET]
```
## Policy Evaluations ##
To get policy evaluations, run:
```
./eval_policy.sh tax3d [TASK_TYPE] [WANDB_CHECKPOINT_RUN_ID] [SEED] [GPU_INDEX]
```
For example, if the previous TAX3D-CD model was trained on a `HangProcCloth` task, one might run:
```
./eval_policy.sh tax3d dedo_proccloth gzc40qe1 1 0
```
Alternatively, if the model was trained for a `HangBag` task, then one should run:
```
./eval_policy.sh tax3d dedo_hangbag gzc40qe1 1 0
```