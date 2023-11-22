import torch
import torch.nn as nn
from VisionTransformer import VisionTransformer
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from tqdm.notebook import tqdm
import numpy as np
import wandb
from HelperFunctions import calculate_accuracy

lr = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 1
BATCH_SIZE = 64


def train(ml_model, learning_rate=lr, image_size=64):
    """
    Train a Vision Transformer model on a custom dataset.
    Args:
        ml_model (VisionTransformer): The Vision Transformer model to train.
        learning_rate (float): Learning rate for training.
        image_size (int): Size to which images are resized.
    """

    # Logging to wandb to track the training
    wandb.init(project="ViT")
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": EPOCHS
    }

    train_transform = transforms.Compose([
        # transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x / 255.0),
        # transforms.Normalize((0.,), (1.,))
    ])

    val_transform = transforms.Compose([
        # transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x / 255.0),
        # transforms.Normalize((0.,), (1.,))
    ])
    # ---
    train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=train_transform)
    validation_dataset = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=val_transform)

    image_to_delete, label_to_delete = train_dataset[0]
    image_2, label_2 = train_dataset[10]

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    ml_model.to(device)

    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []

    print(len(train_dataset), len(validation_dataset))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ml_model.parameters(), lr=learning_rate, weight_decay=1e-3)

    # Training loop
    for epoch in tqdm(range(EPOCHS)):
        ml_model.train()
        total_train, correct_train = 0, 0
        train_loss_epoch = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # print(images.shape)
            # print("yo")

            # Forward pass
            outputs = ml_model(images)
            # _, outputs = torch.max(outputs, dim=1)
            # print(outputs)
            # print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, dim=1)
            correct_train += torch.eq(labels, predicted).sum().item()
            # print(f"correct_train: {correct_train}\ncorrect_train_type: {type(correct_train)}")
            total_train += labels.size(0)
            train_loss_epoch.append(loss.item())
            wandb.log({"batch_train_loss": loss.item(),
                       "batch_train_accuracy": calculate_accuracy(labels, predicted)})

        train_accuracy = 100 * correct_train / total_train
        train_losses.append(np.mean(train_loss_epoch))
        train_accuracies.append(train_accuracy)
        wandb.log({"epoch_train_loss": np.mean(train_loss_epoch),
                   "epoch_train_accuracy": train_accuracy})

        # Validation phase
        ml_model.eval()
        total_val, correct_val = 0, 0
        validation_loss_epoch = []
        with torch.no_grad():
            for val_images, val_labels in validation_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                outputs = ml_model(val_images)
                val_loss = criterion(outputs, val_labels)
                validation_loss_epoch.append(val_loss.item())

                _, val_predicted = torch.max(outputs, dim=1)
                correct_val += torch.eq(val_labels, val_predicted).sum().item()
                total_val += val_labels.size(0)
                if epoch in [10, 20, 30]:
                    print(f"total_val: {total_val}")
                    print(f"V_labels {val_labels}")
                    print(f"v_preds: {val_predicted}")
                    wandb.log({"batch_val_loss": val_loss.item(),
                               "batch_val_accuracy": calculate_accuracy(val_labels, val_predicted)})

        val_accuracy = 100 * correct_val / total_val
        validation_losses.append(np.mean(validation_loss_epoch))
        validation_accuracies.append(val_accuracy)
        wandb.log({"epoch_val_loss": np.mean(validation_loss_epoch),
                   "epoch_val_accuracy": val_accuracy})
        wandb.finish()

    return (train_losses, validation_losses), (train_accuracies, validation_accuracies)
