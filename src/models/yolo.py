import torch
import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        """
        Initialize YOLOv1 Network
        Args:
            S: Grid size
            B: Number of bounding boxes per grid cell
            C: Number of classes
        """
        super(YOLOv1, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(192, 128, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            *[nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1)
            ) for _ in range(4)],
            
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            
            # Final conv layers
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (C + B * 5))
        )
        
        self.S = S
        self.B = B
        self.C = C

    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor of shape (batch_size, 3, 448, 448)
        Returns:
            Tensor of predictions: (batch_size, S, S, C + B * 5)
        """
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        # Reshape to (batch_size, S, S, C + B * 5)
        x = x.view(-1, self.S, self.S, self.C + self.B * 5)
        
        return x