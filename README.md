# Pneumonia Classifier Training (PyTorch)

Training of ResNet-18 fully-connected layer in PyTorch for classifying chest X-ray images into:
- `BACTERIAL_PNEUMONIA`
- `VIRUS_PNEUMONIA`
- `NORMAL`

Repository layout (important files):

- `data_module.py` — dataset / dataloader helpers
- `model.py` — model architecture
- `training.py` — training loop / entrypoint
- `helper_utils.py` — utility functions
- `chest_xray/` — dataset folder with `train/`, `val/`, and `test/` subfolders

Quick start

1. Create a Python virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install minimal dependencies (adjust for CUDA if needed):

```bash
pip install -r requirements.txt
```

3. Train the model (simple invocation):

```bash
python training.py
```

Notes

- Ensure the `chest_xray/` dataset is present and follows the expected train/val folder structure.
- Edit `training.py` or `data_module.py` to change hyperparameters, device (CPU/GPU), or augmentations.

If you'd like, I can add a `requirements.txt`, example commands for evaluation, or a small usage section showing how to load `model_weights.pth`.
