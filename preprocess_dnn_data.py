import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch

def load_and_preprocess_data_dnn(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Strip whitespace from column names
    data.columns = data.columns.str.strip()
    
    # Check column names to ensure 'Label' exists
    print("Columns in the dataset:", data.columns)

    # Ensure 'Label' column exists
    if 'Label' not in data.columns:
        raise KeyError("The 'Label' column is not found in the dataset.")
    
    # Separate features and labels
    X = data.drop('Label', axis=1).values
    y = data['Label'].values

    # Print the shape of the features
    print("Shape of X before preprocessing:", X.shape)
    
    # Encode the labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Print the shape of the training features
    print("Shape of X_train before reshaping:", X_train.shape)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, label_encoder

def create_data_loaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, batch_size=32, val_split=0.2):
    # Create a validation split
    val_size = int(len(X_train_tensor) * val_split)
    train_size = len(X_train_tensor) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        [train_size, val_size]
    )

    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
