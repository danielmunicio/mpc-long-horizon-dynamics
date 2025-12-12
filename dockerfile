FROM python:3.10-slim

# System deps needed for CasADi + l4casadi + SciPy stack
RUN apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade Python build tooling
# IMPORTANT: do NOT install cmake from pip (breaks l4casadi)
RUN pip install --upgrade pip setuptools wheel scikit-build ninja

# ---- PyTorch CPU ----
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# ---- Your dependencies (all CPU-safe) ----
RUN pip install \
    pytorch-lightning==2.6.0 \
    numpy==2.2.6 \
    scipy==1.15.3 \
    pandas==2.3.3 \
    scikit-learn==1.7.2 \
    matplotlib==3.10.7 \
    pillow==12.0.0 \
    tqdm==4.67.1 \
    torchmetrics==1.8.2 \
    lightning-utilities==0.15.2 \
    h5py==3.15.1 \
    networkx==3.3 \
    fsspec==2025.9.0 \
    propcache==0.4.1 \
    python-dateutil==2.9.0.post0 \
    PyYAML==6.0.3 \
    kiwisolver==1.4.9 \
    pyparsing==3.2.5 \
    Jinja2==3.1.6 \
    attrs==25.4.0 \
    six==1.17.0 \
    packaging==25.0 \
    tzdata==2025.2 \
    aiohttp==3.13.2 \
    aiohappyeyeballs==2.6.1 \
    aiosignal==1.4.0 \
    async-timeout==5.0.1 \
    multidict==6.7.0 \
    yarl==1.22.0 \
    idna==3.11

# ---- l4casadi (this should now succeed like before) ----
RUN pip install l4casadi --no-build-isolation

WORKDIR /workspace
CMD ["/bin/bash"]
