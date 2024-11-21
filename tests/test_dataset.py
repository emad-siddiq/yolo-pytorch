import torch
import pytest
import os
import sys
import numpy as np

# Add project root to path
sys.path.append('/content/yolo-pytorch')

from src.data.dataset import YOLODataset
from src.config import Config

class TestYOLODataset:
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing"""
        # Ensure test directories exist
        os.makedirs('/content/yolo-pytorch/data/processed/test', exist_ok=True)
        
        # Create a sample image and label
        sample_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        np.save('/content/yolo-pytorch/data/processed/test/sample_image.npy', sample_img)
        
        # Create a sample label file
        with open('/content/yolo-pytorch/data/processed/test/sample_image.txt', 'w') as f:
            f.write("0 0.5 0.5 0.2 0.2")  # Sample object in center
        
        return YOLODataset(
            img_dir='/content/yolo-pytorch/data/processed/test',
            label_dir='/content/yolo-pytorch/data/processed/test',
            transform=None
        )
    
    def test_dataset_length(self, sample_dataset):
        """Test dataset length"""
        assert len(sample_dataset) > 0
    
    def test_getitem(self, sample_dataset):
        """Test dataset item retrieval"""
        image, label_matrix = sample_dataset[0]
        
        # Check image tensor shape
        assert isinstance(image, torch.Tensor)
        
        # Check label matrix shape and properties
        assert label_matrix.shape == (Config.GRID_SIZE, Config.GRID_SIZE, Config.NUM_CLASSES + 5)
        
        # Check object presence flag
        assert torch.sum(label_matrix[:, :, Config.NUM_CLASSES]) > 0