import torch

class Tester:
    """
    Tester class for evaluating the model on a given test dataset.

    Attributes:
        model (nn.Module): The trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test dataset.
        config (dict): Experiment configuration.
        device (torch.device): Computation device.
    """

    def __init__(self, model, test_loader, config, device):
        """
        Initializes the Tester.

        Args:
            model (nn.Module): The trained model to be tested.
            test_loader (DataLoader): Test data loader.
            config (dict): Experiment configuration.
            device (torch.device): Computation device (CPU or GPU).
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = device

    def test(self):
        """
        Runs the test loop and prints the test accuracy.
        """
        if self.test_loader is None:
            print("No test data provided.")
            return
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
