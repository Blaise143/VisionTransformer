import torch
import matplotlib.pyplot as plt


def calculate_accuracy(y_true, y_pred) -> float:
    """
    Calculate the accuracy of the model
    Args:
        y_true: The true labels
        y_pred: The predicted labels

    Returns: The accuracy of the model given the the arguments

    """
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = (correct / len(y_pred)) * 100
    return accuracy


def plot_loss(losses: tuple, accuracies: tuple) -> None:
    """
    Plots the train and validation losses and accuracies
    Args:
        losses: A tuple of train loss and Validation Loss
        accuracies: A tuple of the train and validation accuracies

    Returns: None
    """
    train_loss, validation_loss = losses
    train_accuracy, validation_accuracy = accuracies

    plt.plot(train_loss, label="Train Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.title("Losses over time")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()

    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(validation_accuracy, label="Validation Accuracy")
    plt.title("Accuracies over time")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
