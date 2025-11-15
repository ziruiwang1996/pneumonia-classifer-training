import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torchvision import models as tv_models
from torchmetrics.classification import Accuracy
import os
from lightning.pytorch.callbacks import EarlyStopping

def load_resnet18(num_classes, weights_path, device=None):
    """
    Initializes a ResNet-18 model, loads weights, and sets it up
    for transfer learning (feature extraction).

    Args:
        num_classes (int): The number of output classes for the new classifier head.
        weights_path (str): The file path to the saved .pth model weights.

    Returns:
        A PyTorch model (ResNet-18) where all layers are frozen except for
        the final classifier head.
    """
    # Initialize a ResNet-18 model without pre-trained weights.
    model = tv_models.resnet18(weights=None)

    # Replace the classifier head to match the number of classes for the new task.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Optionally load a state dictionary (weights) from a local file.
    # If `weights_path` is falsy or the file does not exist, skip loading
    # and return the freshly initialized model. This makes the function
    # robust when a pretrained file isn't available on disk.
    if weights_path:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        map_loc = torch.device(device) if device is not None else torch.device('cpu')
        state_dict = torch.load(weights_path, map_location=map_loc)
        model.load_state_dict(state_dict)

    # Freeze all the parameters in the model.
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze ONLY the parameters of the new classifier head.
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

def define_optimizer_and_scheduler(model, learning_rate, weight_decay):
    """
    Defines the optimizer and learning rate scheduler for the model.

    Args:
        model (nn.Module): The model for which to configure the optimizer.
                           Its parameters will be passed to the optimizer.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay (L2 penalty) for the optimizer.

    Returns:
        A tuple containing the configured optimizer and lr_scheduler.
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2
    )
    
    return optimizer, scheduler

class ChestXRayClassifier(pl.LightningModule):
    """A LightningModule that is focused on tracking validation loss and accuracy."""

    def __init__(self, model_weights_path, num_classes=3, learning_rate=1e-3, weight_decay=1e-2, device=None):
        """
        Initializes the ChestXRayClassifier module.

        Args:
            model_weights_path (str): The file path to the pre-trained ResNet-18 model weights.
            num_classes (int): The number of classes for classification. Defaults to 3.
            learning_rate (float): The learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float): The weight decay (L2 penalty) for the optimizer. Defaults to 1e-2.
        """
        super().__init__()
        # Save all __init__ arguments (model_weights_path, num_classes, etc.) to self.hparams
        self.save_hyperparameters()
        # Load model and, if requested, load weights directly onto the target device.
        self.model = load_resnet18(
            self.hparams.num_classes,
            self.hparams.model_weights_path,
            device=device
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )

        # If a device was provided, move the entire LightningModule to it.
        # Lightning's Trainer will still manage devices when training, but
        # moving here ensures consistent placement if the model is used
        # outside of the Trainer or when weights were loaded to a specific device.
        if device is not None:
            try:
                self.to(torch.device(device))
            except Exception:
                # If moving fails for any reason, continue and let Trainer handle it.
                pass

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of images.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx=None):
        """
        Performs a single training step. Loss calculation is required for backpropagation.

        Args:
        batch (tuple): A tuple containing the input images and their labels.
        batch_idx (int): The index of the current batch. The Lightning Trainer
                         requires this argument, but it's not utilized in this
                         implementation as the logic is the same for all batches.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss

    def validation_step(self, batch, batch_idx=None):
        """
        Performs a single validation step and logs only the loss and accuracy.

        Args:
        batch (tuple): A tuple containing the input images and their labels.
        batch_idx (int): The index of the current batch. The Lightning Trainer
                         requires this argument, but it's not utilized in this
                         implementation as the logic is the same for all batches.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True)

    def configure_optimizers(self):
        """Configures the optimizers and learning rate scheduler."""
        optimizer, scheduler = define_optimizer_and_scheduler(
            self.model,
            self.hparams.learning_rate,
            self.hparams.weight_decay
        ) 
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
    
def early_stopping(num_epochs, stop_threshold):
    """
    Configures and returns a Lightning EarlyStopping callback.

    Args:
        num_epochs (int): The maximum number of epochs, used to set patience.
        stop_threshold (float): The validation accuracy threshold to stop training.

    Returns:
        EarlyStopping: The configured Lightning callback.
    """
    stop = EarlyStopping(
        monitor='val_acc',
        stopping_threshold=stop_threshold,
        patience= num_epochs//2,
        mode='max'
    ) 

    return stop

def run_training(model, data_module, num_epochs, callback, progress_bar=True, dry_run=False):
    """
    Configures and runs a Lightning mixed-precision training process.

    Args:
        model (pl.LightningModule): The model to be trained.
        data_module (pl.LightningDataModule): The data module that provides the datasets.
        num_epochs (int): The maximum number of epochs for training.
        callback (pl.Callback): A callback, such as for early stopping.
        progress_bar (bool): If True, shows the training progress bar. Defaults to True.
        dry_run (bool): If True, runs a quick single batch "dry run" to test the code.
                        Defaults to False.

    Returns:
        A tuple containing:
            - pl.Trainer: The trainer instance after fitting is complete.
            - pl.LightningModule: The trained model with updated weights.
    """
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto", 
        devices=1, 
        precision="16-mixed",
        callbacks=[callback],
        logger=False,
        enable_progress_bar=progress_bar,
        enable_model_summary=False,
        enable_checkpointing=False,
        fast_dev_run=dry_run
    )
    trainer.fit(model, datamodule=data_module)

    return trainer, model