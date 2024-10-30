# Instructions for running this thing on Autobot.


0. Before you do anything, make sure you've built your docker image and pushed it to dockerhub!!!

1. ssh into autobot:

    ```
    ssh <SCS_username>@autobot.vision.cs.cmu.edu
    ```

    a. *YOU ONLY NEED TO DO THIS ONCE*: Add your wandb API key to your bashrc:

        ```bash
        echo 'export WANDB_API_KEY="your_api_key_here"' >> ~/.bashrc
        source ~/.bashrc
        ```

2. Find a node on http://autobot.vision.cs.cmu.edu/mtcmon/ which has open GPUs.

3. SSH into that node:

    ```
    ssh autobot-0-33
    ```

    a. *YOU ONLY NEED TO DO THIS ONCE*: Create some scratch directories for your data and logs.

        ```bash
        mkdir -p /scratch/$(whoami)/data
        mkdir -p /scratch/$(whoami)/logs
        ```


3.5 Put data into `/project_data/held/${whoami}/data' folder, so that you can copy it over to whatever GPU node you are running your job on.

3.6 Put your code into `~/code`

4. Run a training job like so. Don't worry about building or installing. You can modify the files here to map to whatever you want. In future iterations of this, we'll make this easier to do (aka by using a hydra singularity condfig file or something so you don't have to explictly map as arguments).

    You can also change which GPU you want access to using CUDA_VISIBLE_DEVICES below.

    ```bash
    SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 \
    SINGULARITYENV_WANDB_DOCKER_IMAGE=python-ml-project-template \
    singularity exec \
    --nv \
    --pwd /opt/$(whoami)/code \
    -B /scratch/$(whoami)/data:/opt/data \
    -B /scratch/$(whoami)/logs:/opt/logs \
    docker://beisner/python-ml-project-template \
    bash -c "
    [NORMAL TRAINING CALL HERE] cd third_party/3D....../train.sh dp3 dedo_proccloth....
    "
    python scripts/train.py \
        dataset.data_dir=/opt/data \
        log_dir=/opt/logs
    ```



singularity exec --nv \
-B /home/ktsim/code/non-rigid:/opt/eycai/code/non-rigid \
-B /scratch/ktsim/data:/opt/eycai/data \
-B /scratch/ktsim/logs:/opt/eycai/logs \
/scratch/ktsim/singularity/tax3d.sif \
bash -c "cd /opt/eycai/code/non-rigid/third_party/3D-Diffusion-Policy && 
bash scripts/train_policy_goalPC.sh dp3 dedo_proccloth autobot_test 1 0 1 tax3d pointnet 0"