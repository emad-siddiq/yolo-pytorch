import os
import torch
from torch.utils.data import Dataset
import cv2

class YOLODataset(Dataset):
    """Custom Dataset for YOLO"""
    def __init__(self, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        """
        Args:
            img_dir: Directory with images
            label_dir: Directory with label files
            S: Grid size
            B: Number of bounding boxes per grid cell
            C: Number of classes
            transform: Optional image transformation
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
        
        # Find all image files
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, idx):
        """
        Returns preprocessed image and label matrix
        """
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        
        # Replace image extension with txt for labels
        label_filename = os.path.splitext(self.img_files[idx])[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_filename)
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform if specified
        if self.transform:
            image = self.transform(image)
            
        # Process labels to match YOLO format
        label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        
        # Check if label file exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for label in f.readlines():
                    # Expect format: class_id x_center y_center width height
                    class_label, x_center, y_center, width, height = map(float, label.strip().split())
                    
                    # Convert to grid cell coordinates
                    grid_x = int(x_center * self.S)
                    grid_y = int(y_center * self.S)
                    
                    # Cell-relative coordinates
                    x_cell = x_center * self.S - grid_x
                    y_cell = y_center * self.S - grid_y
                    
                    # Cell-relative width and height
                    width_cell = width * self.S
                    height_cell = height * self.S
                    
                    # Only add if this cell doesn't already have an object
                    if label_matrix[grid_y, grid_x, self.C] == 0:
                        # Set class probability
                        label_matrix[grid_y, grid_x, int(class_label)] = 1.0
                        
                        # Set object presence flag and box coordinates
                        label_matrix[grid_y, grid_x, self.C:self.C+5] = torch.tensor([
                            1.0,  # Object present
                            x_cell, 
                            y_cell, 
                            width_cell, 
                            height_cell
                        ])
                    
        return image, label_matrix