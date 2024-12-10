import os
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, plot_dir):
    """
    Plots training and validation loss curves and saves the figure to a file.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch or None if no validation.
        plot_dir (str): Directory to save the plot image.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
    if any(val_losses):
        plt.plot(epochs, val_losses, 'ro-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'loss_curve.png'))
    plt.close()
