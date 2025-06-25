pip check
pip cache purge
conda clean --all -y

rm -rf ~/.cache/pip
rm -rf ~/.cache/torch

conda env update -f environment.yaml --prune
conda remove --name vamba --all -y
conda env create -f environment.yaml
conda activate vamba
pip install flash-attn --no-build-isolation

https://github.com/state-spaces/mamba/releases/tag/v2.2.2/mamba_ssm-2.2.2+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
USE_NINJA=1 MAX_JOBS=16 pip install . --no-build-isolation --no-cache-dir --verbose
USE_NINJA=1 MAX_JOBS=128 pip install . --no-build-isolation --no-cache-dir --verbose



pip list | grep torch
open_clip_torch                   2.31.0
torch                             2.6.0+cu118
torchaudio                        2.6.0+cu118
torchvision                       0.21.0+cu118



wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

pip uninstall -y torch torchvision torchaudio deepspeed transformers flash-attn
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"