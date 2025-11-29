conda create -n torch python=3.11 -y

conda activate torch

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade "sentence-transformers[onnx-gpu]" "transformers==4.53.0" pandas tqdm ipython jupyter notebook ipykernel accelerate

conda install -c nvidia cuda-toolkit -y
pip install flash-attn --no-build-isolation
