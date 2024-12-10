import torch
from torch import nn, optim
from .utils import save_checkpoint
from .visualization import plot_losses
import os

class Trainer:
    """
    Trainer class that manages the training process of a model.

    Attributes:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader or None): DataLoader for the training set.
        val_loader (DataLoader or None): DataLoader for the validation set.
        config (dict): Configuration dictionary.
        device (torch.device): The device to run on (CPU or GPU).
        visualization_enabled (bool): Whether to plot training/validation losses.
        export_loss_enabled (bool): Whether to export the recorded losses to a file.
        train_losses (list): A list of recorded training losses if visualization or export is enabled.
        val_losses (list): A list of recorded validation losses if visualization or export is enabled.
    """

    def __init__(self, model, train_loader, val_loader, config, device):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader or None): Training data loader.
            val_loader (DataLoader or None): Validation data loader.
            config (dict): Experiment configuration.
            device (torch.device): Computation device (CPU or GPU).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.criterion = self._get_loss_function()
        self.optimizer = self._get_optimizer()

        self.visualization_enabled = self.config.get('visualization', {}).get('enabled', False)
        self.export_loss_enabled = self.config.get('export_loss', {}).get('enabled', False)

        # Only initialize loss lists if needed
        if self.visualization_enabled or self.export_loss_enabled:
            self.train_losses = []
            self.val_losses = []

    def _get_loss_function(self):
        """
        Initializes the loss function based on the config.

        Returns:
            nn.Module: A PyTorch loss function.
        """
        loss_config = self.config.get('loss', {})
        loss_type = loss_config['type']
        loss_args = loss_config.get('args', {})
        loss_class = getattr(nn, loss_type)
        return loss_class(**loss_args)

    def _get_optimizer(self):
        """
        Initializes the optimizer based on the config.

        Returns:
            torch.optim.Optimizer: The initialized optimizer.
        """
        optimizer_config = self.config['training']['optimizer']
        optimizer_type = optimizer_config['type']
        optimizer_args = optimizer_config.get('args', {})
        lr = self.config['training']['learning_rate']
        optimizer_class = getattr(optim, optimizer_type)
        return optimizer_class(self.model.parameters(), lr=lr, **optimizer_args)

    def train(self):
        """
        Runs the training loop for the specified number of epochs.
        Records and optionally visualizes or exports losses.
        """
        if self.train_loader is None:
            print("No training data provided.")
            return

        epochs = self.config['training']['epochs']
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_loss:.4f}")

            if self.visualization_enabled or self.export_loss_enabled:
                self.train_losses.append(avg_loss)

            # Validation step if validation loader is provided
            if self.val_loader is not None:
                val_loss = self.validate()
                if self.visualization_enabled or self.export_loss_enabled:
                    self.val_losses.append(val_loss)
                print(f"  Val Loss: {val_loss:.4f}")
            else:
                if self.visualization_enabled or self.export_loss_enabled:
                    self.val_losses.append(None)

            save_checkpoint(self.model, epoch, self.config.get('checkpoint_dir', './checkpoints'))

        # After training, optional visualization and exporting losses
        if self.visualization_enabled:
            plot_dir = self.config['visualization'].get('plot_dir', './plots')
            plot_losses(self.train_losses, self.val_losses, plot_dir)

        if self.export_loss_enabled:
            export_dir = self.config['export_loss'].get('export_dir', './losses')
            self._export_losses(export_dir)

    def validate(self):
        """
        Runs the validation loop to evaluate model performance on the validation set.

        Returns:
            float: The average validation loss over the validation set.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def _export_losses(self, export_dir):
        """
        Exports the recorded training and validation losses to text files.

        Args:
            export_dir (str): Directory to save the loss files.
        """
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        train_path = os.path.join(export_dir, 'train_losses.txt')
        with open(train_path, 'w') as f:
            for l in self.train_losses:
                f.write(f"{l}\n")

        if any(self.val_losses):
            val_path = os.path.join(export_dir, 'val_losses.txt')
            with open(val_path, 'w') as f:
                for l in self.val_losses:
                    f.write(f"{l}\n" if l is not None else "None\n")
