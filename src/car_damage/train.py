from pathlib import Path
import copy
import time

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

def set_batchnorm_eval(model):
    """
    Keep BatchNorm running statistics fixed during training.
    Trainable BatchNorm parameters can still receive gradients.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

def get_device():
    """Select a supported training device."""
    return torch.device("cpu")

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one complete epoch."""

    model.train()

    set_batchnorm_eval(model)

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(loader, desc="Training")

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        # Clear the gradients from previous batch
        optimizer.zero_grad()

        # Forward pass
        output = model(images)

        # Claculate the Loss
        loss = criterion(output, labels)

        # Backward pass
        loss.backward()

        # Updates the weights
        optimizer.step()

        # Calculate metrics
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = torch.argmax(output, dim=1)
        correct_predictions += (predictions==labels).sum().item()
        total_samples += batch_size

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss/total_samples
    epoch_accuracy = correct_predictions/total_samples

    return epoch_loss, epoch_accuracy

def evaluate_one_epoch(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        progress_bar = tqdm(loader, desc="Validation")

        for images, labels in progress_bar:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size

            predictions = torch.argmax(outputs, dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    epoch_loss = running_loss / total_samples

    epoch_accuracy = correct_predictions / total_samples

    epoch_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return epoch_loss, epoch_accuracy, epoch_macro_f1



def train_model(
    model,
    train_loader,
    val_loader,
    epochs=5,
    patience=2,
    save_path="artifacts/best_model_exp4.pth",
    class_weights=None,
    layer3_lr=1e-5,
    layer4_lr=1e-5,
    classifier_lr=1e-4,
    weight_decay=1e-4
):

    device = get_device()
    model = model.to(device)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Use only trainable parameters
    optimizer = torch.optim.Adam(
    [
        {
            "params": model.layer3.parameters(),
            "lr": layer3_lr
        },
        {
            "params": model.layer4.parameters(),
            "lr": layer4_lr
        },
        {
            "params": model.fc.parameters(),
            "lr": classifier_lr
        }
    ],
    weight_decay=weight_decay)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_macro_f1 = -1.0

    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": []
    }

    for epoch in range(epochs):

        start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # Validation
        val_loss, val_accuracy, val_macro_f1 = (
            evaluate_one_epoch(
                model,
                val_loader,
                criterion,
                device
            )
        )

        epoch_time = time.time() - start_time

        # Store metrics
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_macro_f1"].append(val_macro_f1)

        # Display metrics
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.4f}")

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Macro F1: {val_macro_f1:.4f}")

        print(f"Epoch Time: {epoch_time / 60:.2f} minutes")

        # Save checkpoint based on macro F1
        if val_macro_f1 > best_val_macro_f1:

            best_val_macro_f1 = val_macro_f1

            best_model_weights = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "model_state_dict": best_model_weights,
                    "best_val_macro_f1": best_val_macro_f1,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "epoch": epoch + 1
                },
                save_path
            )

            epochs_without_improvement = 0

            print("Best macro F1 checkpoint saved!")

        else:

            epochs_without_improvement += 1

            print(
                f"No macro F1 improvement: "
                f"{epochs_without_improvement}/{patience}"
            )

        if epochs_without_improvement >= patience:

            print("Early stopping triggered!")
            break

    # Restore best model weights
    model.load_state_dict(best_model_weights)

    print("\nTraining completed!")
    print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
    print(f"Best model saved at: {save_path}")

    return model, history


