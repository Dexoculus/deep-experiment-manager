# Pytorch Experiment Manager

Pytorch Experiment Manager is a flexible and modular framework for training, testing, and managing deep learning experiments in PyTorch. It provides a clear structure for organizing your models, datasets, transforms, and training pipelines, enabling easy configuration and experimentation with different settings.

## Features

- **Config-Driven Setup**: All parameters (model, dataset, training, testing, loss, preprocessing, visualization, and exporting losses) are configured via a YAML file, making experiments reproducible and easily adjustable.
- **Dynamic Loading**: Models, datasets, and preprocessing functions are loaded dynamically from user-specified modules, allowing you to integrate your own code without modifying the core framework.
- **Task Agnostic**: Supports various tasks such as classification, regression, or generation by specifying different models, losses, and transforms.
- **Optional Features**:
  - **Visualization**: Plot training and validation loss curves and save them as images.
  - **Exporting Losses**: Save recorded losses to text files for further analysis.
- **Clear and Modular Code Structure**: Separation of concerns into modules (`manager`, `trainer`, `tester`, `datasets`, `utils`, etc.) for improved maintainability and scalability.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ExperimentManager.git
cd ExperimentManager
```
2. Install required dependencies:
```bash
pip install -r requirements.txt
```
Make sure PyTorch and other dependencies (such as torchvision, yaml, matplotlib) are installed. Adjust the requirements as needed.

## Usage

1. Edit the config.yaml:
Update paths, model modules, dataset classes, preprocessing functions, training parameters, and so forth to match your project.

model modules, dataset classes, preprocessing functions must be defined first. 
You can also check the example code in `/test` directory. 
Example:
```yaml
  model:
  module: 'your.model'
  class: 'YourModel'
  args:
    input_size: 784
    output_size: 10 # If Classification task, than number of classes

training:
  epochs: 10
  learning_rate: 0.001
  optimizer:
    type: 'Adam'
    args:
      weight_decay: 0.0001

loss:
  type: 'CrossEntropyLoss'
  args: {}

dataset:
  module: 'your.dataset'
  class: 'YourDataset'
  args:
    preprocessing:
      module: 'your.preprocessing_module'
      function: 'example_preprocessing'
      args:
        normalize: True
        augment: False
    train:
      args:
        root: 'your data root directory'
        train: True
        download: True
      loader:
        batch_size: 64
        shuffle: True
    valid:
      args:
      loader:
    test:
      args:
        root: 'your data root directory'
        train: True
        download: True
      loader:
        batch_size: 1000
        shuffle: False

visualization:
  enabled: True
  plot_dir: './plots'

export_loss:
  enabled: True
  export_dir: './losses'
```

2. Write the Model Experiment Code:
For example, this would be code for main.py

```python
import argparse
from experiment_manager import ExperimentManager

def main():
    """
    Main entry point for running the experiment.

    This script will:
    - Parse command-line arguments.
    - Initialize and run the ExperimentManager with the given config file.
    """
    parser = argparse.ArgumentParser(description="Run experiment using ExperimentManager.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # Initialize ExperimentManager with the given config
    manager = ExperimentManager(config_path=args.config)
    # Run the full experiment (training and optional testing)
    manager.run_experiment()

if __name__ == '__main__':
    main()
```

3. Run the main code:
```bash
python main.py --config.yaml
```
in this case, config.yaml must be placed in root directory.

this will:
- Load the specified model, dataset, and transforms.
- Run training for the specified number of epochs.
- Optionally validate after each epoch.
- Save checkpoints, plot loss curves, and export losses if enabled.

4. Check Outputs:
- Checkpoints: Stored in `./checkpoints/` by default.
- Loss plots: Stored in `./plots/` if visualization is enabled.
- Loss files: Stored in `./losses/` if exporting is enabled.
- Logs: Training and validation logs displayed in the terminal.

## License
This project is provided under the MIT License.


