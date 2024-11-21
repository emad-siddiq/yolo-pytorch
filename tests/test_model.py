import torch
import pytest
import sys

# Add project root to path
sys.path.append('/content/yolo-pytorch')

from src.models.yolo import YOLOv1
from src.config import Config

class TestYOLOModel:
    @pytest.fixture
    def model(self):
        """Create a YOLO model for testing"""
        return YOLOv1(S=Config.GRID_SIZE, B=Config.NUM_BOXES, C=Config.NUM_CLASSES)
    
    def test_model_output_shape(self, model):
        """Test model output shape"""
        # Create a random input tensor
        x = torch.randn(1, 3, 448, 448)
        
        # Get model output
        output = model(x)
        
        # Check output shape
        expected_shape = (1, Config.GRID_SIZE, Config.GRID_SIZE, Config.NUM_CLASSES + Config.NUM_BOXES * 5)
        assert output.shape == expected_shape
    
    def test_model_parameters(self, model):
        """Test model parameter count"""
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
    
    def test_model_forward_pass(self, model):
        """Test model forward pass"""
        # Create a random input tensor
        x = torch.randn(4, 3, 448, 448)  # Batch of 4 images
        
        # Ensure no errors during forward pass
        try:
            output = model(x)
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")
        
        assert output is not None