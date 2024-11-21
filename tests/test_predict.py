import torch
import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append('/content/yolo-pytorch')

from src.utils.predict import YOLOPredictor
from src.config import Config

class TestYOLOPredictor:
    @pytest.fixture
    def predictor(self):
        """Create a mock predictor for testing"""
        # Ensure weights directory exists
        os.makedirs(Config.WEIGHTS_DIR, exist_ok=True)
        
        # Create a dummy model state dict
        dummy_state_dict = {
            'conv_layers.0.weight': torch.randn(64, 3, 7, 7),
            'fc_layers.1.weight': torch.randn(4096, 1024 * Config.GRID_SIZE * Config.GRID_SIZE)
        }
        
        # Save dummy weights
        dummy_weights_path = os.path.join(Config.WEIGHTS_DIR, 'test_model.pth')
        torch.save(dummy_state_dict, dummy_weights_path)
        
        return YOLOPredictor(
            model_path=dummy_weights_path, 
            S=Config.GRID_SIZE, 
            B=Config.NUM_BOXES, 
            C=Config.NUM_CLASSES
        )
    
    def test_preprocessor(self, predictor):
        """Test image preprocessing"""
        # Create a sample image
        sample_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        try:
            image_tensor, orig_size = predictor.preprocess_image(sample_image)
        except Exception as e:
            pytest.fail(f"Image preprocessing failed: {e}")
        
        # Check tensor shape and properties
        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == (1, 3, 448, 448)
        assert len(orig_size) == 2
    
    def test_nms(self, predictor):
        """Test non-maximum suppression"""
        # Mock bounding boxes, labels, and confidences
        bboxes = np.array([
            [10, 10, 50, 50],   # First box
            [20, 20, 60, 60],   # Overlapping box
            [200, 200, 250, 250]  # Non-overlapping box
        ])
        labels = np.array([0, 0, 1])
        confidences = np.array([0.9, 0.8, 0.7])
        
        # Apply NMS
        filtered_bboxes, filtered_labels, filtered_confidences = predictor.non_max_suppression(
            bboxes, labels, confidences
        )
        
        # Check that NMS reduced number of boxes
        assert len(filtered_bboxes) <= len(bboxes)
        assert len(filtered_labels) == len(filtered_bboxes)
        assert len(filtered_confidences) == len(filtered_bboxes)