import sys
import os
import importlib
import torch

from .config import load_config
from .trainer import Trainer
from .tester import Tester
from .datasets import get_datasets
from .utils import set_seed


class ExperimentManager:
    """
    ExperimentManager orchestrates the entire experiment workflow:
    - Loads configuration
    - Initializes model, datasets, and preprocessing
    - Sets up training and testing
    - Executes the training and (optionally) testing phases
    """

    def __init__(self, config_path):
        """
        Initializes the ExperimentManager.

        Args:
            config_path (str): The path to the configuration file.
        """
        self.config = load_config(config_path)
        self.device = self._get_device()

    def _get_device(self):
        """
        Determines the computation device (CPU or GPU).

        Returns:
            torch.device: The device to use.
        """
        use_cuda = True  # Optionally, this can be derived from config if needed
        return torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    def run_experiment(self):
        """
        Runs the full experiment including training and optional testing.
        """
        set_seed(self.config.get('seed', 42))

        # Load model and move to appropriate device
        model = self._load_model().to(self.device)

        # Retrieve train, valid, and test loaders based on config
        train_loader, valid_loader, test_loader = get_datasets(self.config['dataset'])

        # Initialize trainer and run training
        trainer = Trainer(model, train_loader, valid_loader, self.config, self.device)
        trainer.train()

        # If a test loader is available, run testing
        if test_loader is not None:
            tester = Tester(model, test_loader, self.config, self.device)
            tester.test()

    def _load_model(self):
        """
        Dynamically loads and instantiates the model specified in the config.

        Returns:
            nn.Module: The instantiated PyTorch model.
        """
        model_config = self.config['model']
        module_name = model_config['module']
        class_name = model_config['class']
        model_args = model_config.get('args', {})

        sys.path.append(os.getcwd())
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        model = model_class(**model_args)
        return model
