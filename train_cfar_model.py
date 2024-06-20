import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm  # Add tqdm for progress bar
from preprocess_data import load_and_preprocess_data, create_data_loaders
from cifar_model import CIFAR_CNN
import os

# Path to save the model in the current directory
MODEL_PATH = 'cifar_cnn.pth'

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=5):
    early_stopping = EarlyStopping(patience=patience)
    model.train()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')  # Add progress bar

        for inputs, labels in progress_bar:  # Use progress bar
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=running_loss / len(train_loader.dataset))  # Update progress bar

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = validate(model, val_loader, criterion)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Save the model after training
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    return val_loss

def main():
    # Load and preprocess data
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, label_encoder = load_and_preprocess_data('cleaned_cicids2017.csv')
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, val_split=0.2)

    # Initialize the model
    model = CIFAR_CNN(num_classes=len(label_encoder.classes_))
    model.num_classes = len(label_encoder.classes_)  # Ensure num_classes is set
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model with early stopping
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=5)

if __name__ == "__main__":
    main()
