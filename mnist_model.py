import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self, num_classes=11):
        super(MNIST_CNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout2d(p=0.25)
        
        self.fc1 = nn.Linear(64 * 2 * 2, 128)  # Adjust based on input size after convolutions and pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output: (batch_size, 32, 4, 4)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (batch_size, 64, 2, 2)
        x = self.dropout(x)
        
        x = x.view(-1, 64 * 2 * 2)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

# Note: Ensure your training script imports this updated model class and uses it correctly.
