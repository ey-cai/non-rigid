# Use the official Ubuntu 20.04 image as the base
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y curl git build-essential libssl-dev zlib1g-dev libbz2-dev \
    git \
    libreadline-dev libsqlite3-dev wget llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pyenv
ENV CODING_ROOT="/opt/eycai"

WORKDIR $CODING_ROOT
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv

ENV PYENV_ROOT="$CODING_ROOT/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# Install Python 3.10 using pyenv
RUN pyenv install 3.9
RUN pyenv global 3.9

# Install specific version of pip and setuptools
RUN pip install --upgrade pip==23.3.1 setuptools==65.5

# Install PyTorch with CUDA support (make sure to adjust this depending on your CUDA version)
# RUN pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
# RUN pip install torch_geometric
# RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
# RUN pip install fvcore iopath
# RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/download.html
# RUN pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html
# RUN pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html

# Make the working directory the home directory
RUN mkdir $CODING_ROOT/code

# # Get R-PAD dependencies
# WORKDIR $CODING_ROOT/code
# RUN git clone https://github.com/r-pad/pyg_libs.git
# WORKDIR $CODING_ROOT/code/pyg_libs
# RUN pip install -e .

# WORKDIR $CODING_ROOT/code
# RUN git clone https://github.com/r-pad/visualize_3d.git
# WORKDIR $CODING_ROOT/code/visualize_3d
# RUN git checkout modified_plots
# RUN pip install -e .

# Only copy in the source code that is necessary for the dependencies to install
WORKDIR $CODING_ROOT/code/non-rigid
COPY ./src $CODING_ROOT/code/non-rigid/src
COPY ./third_party $CODING_ROOT/code/non-rigid/third_party
COPY ./setup.py $CODING_ROOT/code/non-rigid/setup.py
COPY ./pyproject.toml $CODING_ROOT/code/non-rigid/pyproject.toml
COPY ./requirements-gpu.txt $CODING_ROOT/code/non-rigid/requirements-gpu.txt

# Install torch/additional GPU dependencies
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Install third-party dependencies
RUN cd third_party/dedo && \
    pip install --no-cache-dir -e . &&  \
    cd ../..
RUN cd third_party/3D-Diffusion-Policy/3D-Diffusion-Policy && \
    pip install --no-cache-dir -e . && \
    cd ../../..

# Install the non-rigid package
RUN pip install --no-cache-dir -e .

# Changes to the configs and scripts will not require a rebuild
COPY ./configs $CODING_ROOT/code/non-rigid/configs
COPY ./scripts $CODING_ROOT/code/non-rigid/scripts

RUN git config --global --add safe.directory /root/code/non-rigid

# Make a data directory.
RUN mkdir $CODING_ROOT/data

# Make a logs directory.
RUN mkdir $CODING_ROOT/logs

# Set up the entry point
# CMD ["python", "-c", "import torch; print(torch.cuda.is_available())"]
