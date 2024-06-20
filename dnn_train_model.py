import torch
import torch.nn as nn
import torch.optim as optim
from preprocess_dnn_data import load_and_preprocess_data_dnn, create_data_loaders
from dnn_model import DNN

# Directory to save the model
MODEL_PATH = 'dnn_model.pth'

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    best_model_wts = None

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping")
            break

        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        val_loss, _ = validate(model, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
            print(f'Validation loss improved at epoch {epoch + 1} with validation loss {val_loss:.4f}')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True

    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    # Save the final model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Model saved after training completion.')

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    val_loss /= len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
    return val_loss, accuracy

def main():
    # Load and preprocess data
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, label_encoder = load_and_preprocess_data_dnn('cleaned_cicids2017.csv')
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)

    # Initialize the model
    input_size = X_train_tensor.shape[1]
    model = DNN(input_size=input_size, num_classes=len(label_encoder.classes_))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model with early stopping
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, patience=5)

if __name__ == "__main__":
    main()
