import torch
from model import ChestXRayClassifier, early_stopping, run_training
from data_module import ChestXRayDataModule
import os
import lightning.pytorch as pl

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    print(f"Using device: {device}")

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "chest_xray"))

    pretrained_weights = "./resnet18_chest_xray_classifier_weights.pth"
    # If the weights file doesn't exist, run with randomly initialized weights.
    if not os.path.isfile(pretrained_weights):
        print(f"Weights file not found at {pretrained_weights}; continuing without pretrained weights.")
        pretrained_weights = None

    early_stopping_callback = early_stopping(num_epochs=10, stop_threshold=0.85)

    # Set num_workers=0 for safe startup on macOS / spawn-based multiprocessing.
    dm = ChestXRayDataModule(data_dir, device=device, num_workers=0)
    dm.setup()

    # Create model and ensure it's aware of the target device. The Lightning Trainer
    # will still manage device placement during `fit`, but passing `device` helps
    # load weights directly onto the correct device and keeps behavior consistent
    # when using the model outside of Trainer.
    model = ChestXRayClassifier(model_weights_path=pretrained_weights, device=device)

    trained_trainer, trained_model = run_training(
        model, dm, num_epochs=10, callback=early_stopping_callback
    )
    # Print a message to confirm that the training has finished.
    print("\n--- Training Complete ---")

    # Get the final metrics from the trainer object
    final_metrics = trained_trainer.callback_metrics
    # Extract the validation accuracy and convert it to a number
    final_val_acc = final_metrics["val_acc"].item()
    # Print the final validation accuracy
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")

    fc_state = {k: v.cpu() for k, v in trained_model.model.fc.state_dict().items()}
    torch.save(fc_state, "resnet18_fc_only.pth")


if __name__ == '__main__':
    # On platforms that use 'spawn' to create child processes (macOS by default),
    # protect the entrypoint to avoid multiprocessing re-import issues.
    import multiprocessing as _mp
    try:
        _mp.freeze_support()
    except Exception:
        pass
    main()