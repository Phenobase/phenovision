#!/bin/bash
# Build the dedicated `two_noise` conda env for the two-noise subproject.
# Strand 1 (PyTorch) + Strand 2 (JAX, GPU-capable) in one env; we run a GPU
# coexistence smoke test after this and split into two envs only if they clash.
#
# Run from a login node (CPU). GPU verification happens later via a SLURM smoke job.
set -euo pipefail

ENV=two_noise
echo "=== [1/4] create env with conda-forge base (no GPU libs yet) ==="
mamba create -y -n "$ENV" -c conda-forge \
    python=3.11 \
    numpy scipy pandas matplotlib \
    pytest \
    einops \
    hydra-core omegaconf \
    pyyaml tqdm \
    scikit-learn

echo "=== [2/4] activate and locate pip ==="
source /home/${USER}/.bashrc
source activate "$ENV"
which python pip
python --version

echo "=== [3/4] PyTorch (cu124) then JAX (cuda12) via pip wheels ==="
# PyTorch first (matches the working cu124 stack in reticulate-gpu2).
pip install --no-input torch torchvision --index-url https://download.pytorch.org/whl/cu124
# JAX with bundled CUDA 12 (same major as torch -> best chance of coexistence).
pip install --no-input "jax[cuda12]"

echo "=== [4/4] remaining Python deps ==="
# numpyro needs jax (installed above); laplace-torch + pyhessian need torch.
pip install --no-input \
    numpyro \
    laplace-torch \
    pyhessian \
    cma \
    timm transformers \
    safetensors

echo "=== DONE: env $ENV built ==="
python - <<'PY'
import importlib
for m in ["numpy","scipy","pandas","matplotlib","torch","jax","numpyro","laplace","pyhessian","cma","timm","transformers","einops","hydra"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"FAIL {m:14s} {type(e).__name__}: {e}")
PY
