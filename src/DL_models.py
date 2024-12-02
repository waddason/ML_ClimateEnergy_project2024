"""
DL_models.py

This module contains the definition of Deep Learning models for the ML_ClimateEnergy_project2024.

Author: Tristan Waddington
Date: dec 2024
"""

###############################################################################
# Imports
###############################################################################
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
import torch.optim as optim

from pathlib import Path

# constants
saved_models_dir = Path("saved_models")

###############################################################################
# Functions
###############################################################################


# -----------------------------------------------------------------------------
# 1. Tensor creation
# -----------------------------------------------------------------------------
def create_data_loader(
    X: pd.DataFrame, y: pd.DataFrame, batch_size: int, shuffle: bool = True
):
    """
    Create a torch data loader from the X and y DataFrames.
    """
    # Convert the DataFrames to tensors
    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)

    # Create the dataset
    dataset = TensorDataset(X, y)
    # create the DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


# -----------------------------------------------------------------------------
# 2. Model training
# -----------------------------------------------------------------------------
def train_model(
    model, train_loader, val_loader, criterion, optimizer, n_epochs=10, print_every_n=1
):
    """
    Train a model on a dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    train_loader : torch.utils.data.DataLoader
        The training dataset.
    val_loader : torch.utils.data.DataLoader
        The validation dataset.
    criterion : torch.nn.Module
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer.
    n_epochs : int, optional
        The number of epochs to train the model, by default 10.
    print_every_n : int, optional
        Print the losses every n epochs, by default 1.
    Returns
    -------
    train_losses : list
        The training losses.
    test_losses : list
        The test losses.
    """
    train_losses = []
    test_losses = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation loss on each epoch
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                y_pred = model(X)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        test_losses.append(val_loss)

        # Print the losses each n epochs
        if (epoch + 1) % print_every_n == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}"
            )

    return train_losses, test_losses


# -----------------------------------------------------------------------------
# 3. Model evaluation
# -----------------------------------------------------------------------------
def evaluate_model(model, loader, criterion):
    """
    Evaluate the model on the loader data.
    """
    model.eval()
    losses = []
    y_pred = torch.empty(0)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            losses.append(loss.item())
            y_pred = torch.cat((y_pred, output), dim=0)
    return y_pred, losses


# -----------------------------------------------------------------------------
# 3. Save the Model
# -----------------------------------------------------------------------------
def save_model(model: nn.Module, model_name: str):
    """Save the model to the disk"""
    torch.save(model.state_dict(), saved_models_dir / f"{model_name}.pt")
    print(f"Model saved as models/{model_name}.pt")


# -----------------------------------------------------------------------------
# 4. Load the Model
# -----------------------------------------------------------------------------
def load_model(model, model_name: str) -> nn.Module:
    """Load the model from the disk"""
    model.load_state_dict(
        torch.load(saved_models_dir / f"{model_name}.pt", weights_only=True)
    )
    model.eval()
    print(f"Model loaded from models/{model_name}.pt")
    return model


###############################################################################
# Models
###############################################################################


# ------------------------------------------------------------------------------
# 1. a GRU model
# ------------------------------------------------------------------------------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.1
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x.unsqueeze(1), h0)
        out = self.fc(out[:, -1, :])
        return out


# ------------------------------------------------------------------------------
# 2. a CNN model
# ------------------------------------------------------------------------------
class Conv1DModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        x = torch.relu(self.conv3(x))
        x = self.dropout(x)
        x = torch.max_pool1d(x, kernel_size=x.size(2))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
