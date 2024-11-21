import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.models.yolo import YOLOv1
from src.models.loss import YOLOLoss
from src.data.dataset import YOLODataset
from src.config import Config

def create_transforms():
    """Create image transformations for training"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x/255.0)
    ])

def train_yolo(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    """
    Train YOLO model with validation
    
    Args:
        model: YOLO model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epochs: Number of training epochs
        device: Training device (cuda/cpu)
    """
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print training progress
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"Training Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images)
                loss = criterion(predictions, labels)
                val_loss += loss.item()
        
        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, f"{Config.WEIGHTS_DIR}/best_model.pth")
        
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Training Loss: {train_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}")
    
    # Final model save
    torch.save(model.state_dict(), f"{Config.WEIGHTS_DIR}/final_model.pth")

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = YOLOv1(S=Config.GRID_SIZE, 
                   B=Config.NUM_BOXES, 
                   C=Config.NUM_CLASSES).to(device)
    
    # Transforms
    transform = create_transforms()
    
    # Datasets and DataLoaders
    train_dataset = YOLODataset(
        img_dir=Config.TRAIN_DIR,
        label_dir=Config.TRAIN_DIR,
        transform=transform
    )
    
    val_dataset = YOLODataset(
        img_dir=Config.VAL_DIR,
        label_dir=Config.VAL_DIR,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    # Loss and Optimizer
    criterion = YOLOLoss(S=Config.GRID_SIZE, 
                         B=Config.NUM_BOXES, 
                         C=Config.NUM_CLASSES)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=3, 
        verbose=True
    )
    
    # Train
    train_yolo(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler,
        epochs=Config.EPOCHS, 
        device=device
    )

if __name__ == "__main__":
    main()