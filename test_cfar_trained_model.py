import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm  # Add tqdm for progress bar
from preprocess_data import load_and_preprocess_data, create_data_loaders
from cifar_model import CIFAR_CNN  # Ensure this is the correct path
import os
import nbformat  # Import nbformat directly

# Path to the saved model in the current directory
MODEL_PATH = 'cifar_cnn.pth'

# Mapping of numeric labels to attack types
attack_type_mapping = {
    0: 'Benign',
    1: 'DDoS',
    2: 'DoS Hulk',
    3: 'DoS GoldenEye',
    4: 'DoS Slowloris',
    5: 'DoS Slowhttptest',
    6: 'Bot',
    7: 'Infiltration',
    8: 'PortScan',
    9: 'Brute Force -Web',
    10: 'Brute Force -XSS',
    11: 'SQL Injection',
    12: 'SSH-Patator',
    13: 'FTP-Patator',
    14: 'Heartbleed'
}

def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Evaluating')  # Add progress bar
        for inputs, labels in progress_bar:  # Use progress bar
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss = running_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    report = classification_report(all_labels, all_predictions, target_names=[attack_type_mapping[i] for i in range(model.num_classes)], output_dict=True, zero_division=0)
    precision = report['weighted avg']['precision'] * 100
    recall = report['weighted avg']['recall'] * 100
    f1_score = report['weighted avg']['f1-score'] * 100

    results = {
        'Test Loss': test_loss,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Classification Report': classification_report(all_labels, all_predictions, target_names=[attack_type_mapping[i] for i in range(model.num_classes)], zero_division=0)
    }

    # Print performance by attack types
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        indices = [i for i, x in enumerate(all_labels) if x == label]
        attack_labels = [all_labels[i] for i in indices]
        attack_predictions = [all_predictions[i] for i in indices]

        attack_accuracy = accuracy_score(attack_labels, attack_predictions) * 100
        attack_report = classification_report(attack_labels, attack_predictions, output_dict=True, zero_division=0)
        attack_precision = attack_report['weighted avg']['precision'] * 100
        attack_recall = attack_report['weighted avg']['recall'] * 100
        attack_f1_score = attack_report['weighted avg']['f1-score'] * 100

        attack_name = attack_type_mapping.get(label, f'Unknown Attack ({label})')
        print(f'\nPerformance for attack type {attack_name}:')
        print(f'Accuracy: {attack_accuracy:.4f}%, Precision: {attack_precision:.4f}%, Recall: {attack_recall:.4f}%, F1-Score: {attack_f1_score:.4f}%')

    return results

def save_results_to_ipynb(results):
    cells = [
        nbformat.v4.new_markdown_cell("# Model Evaluation Results"),
        nbformat.v4.new_code_cell(
            f"results = {results}\n\n"
            f"print('Test Loss: {results['Test Loss']:.4f}')\n"
            f"print('Accuracy: {results['Accuracy']:.2f}%')\n"
            f"print('Precision: {results['Precision']:.2f}%')\n"
            f"print('Recall: {results['Recall']:.2f}%')\n"
            f"print('F1-Score: {results['F1-Score']:.2f}%')\n"
            f"print('\\nDetailed classification report:\\n' + results['Classification Report'])"
        )
    ]

    nb = nbformat.v4.new_notebook(cells=cells)
    with open('test_cifar_CNN-4_model.ipynb', 'w') as f:
        nbformat.write(nb, f)
    print("Results saved to test_cifar_CNN-4_model.ipynb")

def main():
    # Load and preprocess data
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, label_encoder = load_and_preprocess_data('cleaned_cicids2017.csv')
    
    # Create data loaders (train_loader, val_loader, and test_loader)
    train_loader, val_loader, test_loader = create_data_loaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, batch_size=32, val_split=0.2)

    # Initialize the model
    model = CIFAR_CNN(num_classes=len(label_encoder.classes_))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the saved model weights
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f'Model loaded from {MODEL_PATH}')

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    results = evaluate(model, test_loader, criterion)

    # Save results to a Jupyter notebook
    save_results_to_ipynb(results)

if __name__ == "__main__":
    main()
