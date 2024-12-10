import importlib
import sys
import os

def get_preprocessing(preproc_config):
    """
    Dynamically loads a preprocessing function or class specified in the config.

    Args:
        preproc_config (dict or None): A dictionary specifying the preprocessing function or None.

    Returns:
        callable or None: A preprocessing callable that takes (sample) as input, or None if not specified.
    """
    if preproc_config is None:
        return None
    module_name = preproc_config['module']
    func_name = preproc_config['function']
    func_args = preproc_config.get('args', {})

    sys.path.append(os.getcwd())
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)

    # Assume func is a callable constructor returning a transform
    # If func_args is empty and func is a callable that needs no args, just call func()
    transform = func(**func_args) if func_args else func()
    return transform

def get_datasets(dataset_config):
    """
    Creates train, valid, and test loaders based on the dataset configuration.

    The config should specify a dataset class and arguments for train/valid/test splits,
    including preprocessing and DataLoader parameters.

    Args:
        dataset_config (dict): Configuration dictionary for dataset and loaders.

    Returns:
        tuple: (train_loader, valid_loader, test_loader) or (None, None, None) if not specified.
    """
    module_name = dataset_config['module']
    class_name = dataset_config['class']
    ds_args = dataset_config.get('args', {})

    # Extract preprocessing configuration and load transform
    preproc_config = ds_args.pop('preprocessing', None)
    transform_func = get_preprocessing(preproc_config)

    sys.path.append(os.getcwd())
    module = importlib.import_module(module_name)
    dataset_class = getattr(module, class_name)

    from torch.utils.data import DataLoader

    def create_dataset_and_loader(split_name):
        """
        Creates a dataset and loader for a given split (train/valid/test).

        Args:
            split_name (str): The name of the dataset split ('train', 'valid', 'test').

        Returns:
            tuple: (dataset, loader) or (None, None) if not available.
        """
        if split_name not in ds_args:
            return None, None
        split_config = ds_args[split_name]
        if split_config is None:
            return None, None

        split_dataset_args = split_config.get('args', {})
        split_dataset_args['transform'] = transform_func
        dataset = dataset_class(**split_dataset_args)

        loader_args = split_config.get('loader', None)
        if loader_args is not None and len(loader_args) > 0:
            loader = DataLoader(dataset=dataset, **loader_args)
        else:
            loader = None
        return dataset, loader

    train_dataset, train_loader = create_dataset_and_loader('train')
    valid_dataset, valid_loader = create_dataset_and_loader('valid')
    test_dataset, test_loader = create_dataset_and_loader('test')

    return train_loader, valid_loader, test_loader
