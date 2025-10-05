FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 


# Install common tools
RUN apt update && apt install -y \
    wget git curl vim tmux nano sudo build-essential git-lfs iputils-ping net-tools \
    rdma-core ibverbs-utils perftest librdmacm-dev libibverbs-dev aria2 

RUN DEBIAN_FRONTEND=noninteractive  apt install -y tzdata cmake
# Install CUDA toolkit (includes nvcc)
RUN apt update && apt install -y \
    cuda-toolkit-12-4 \
    cuda-command-line-tools-12-4 \
    cuda-libraries-12-4 \
    cuda-libraries-dev-12-4

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && rm miniconda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# Set environment variables
ENV TORCHDYNAMO_DISABLE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=$PYTHONPATH:/workspace

# Ensure nvcc is in PATH
ENV PATH=/usr/local/cuda-12.4/bin:${PATH}

# Copy http_terminal_server.py to root directory
COPY http_terminal_server.py /http_terminal_server.py

RUN conda init
# Install dependencies for http_terminal_server
RUN pip install ptyprocess flask


# Set working directory
WORKDIR /workspace

# Start http_terminal_server and activate conda environment automatically
RUN echo '' >> ~/.bashrc && \
    echo '# Auto-setup http_terminal_server' >> ~/.bashrc && \
    echo 'if ! pgrep -f "http_terminal_server.py" > /dev/null; then' >> ~/.bashrc && \
    echo '    nohup python /http_terminal_server.py > /tmp/http_terminal_server.log 2>&1 &' >> ~/.bashrc && \
    echo '    echo "Started http_terminal_server in background"' >> ~/.bashrc && \
    echo 'fi' >> ~/.bashrc && \
    echo 'if [ -d "/workspace" ]; then' >> ~/.bashrc && \
    echo '' >> ~/.bashrc && \
    echo '    # Auto-setup http_terminal_server and activate conda environment (If exists)' >> ~/.bashrc && \
    echo '    if [ -d "/workspace/conda" ]; then' >> ~/.bashrc && \
    echo '        conda activate /workspace/conda' >> ~/.bashrc && \
    echo '    fi' >> ~/.bashrc && \
    echo '' >> ~/.bashrc && \
    echo '    # Enter /workspace directory' >> ~/.bashrc && \
    echo '    cd /workspace && ls > /dev/null' >> ~/.bashrc && \
    echo 'fi' >> ~/.bashrc
