import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import urllib.request
from pathlib import Path
import io

class YOLOImagePreprocessor:
    """
    Handles image preprocessing for YOLO model input
    """
    def __init__(self, input_size=448):
        self.input_size = input_size

        # Standard transform pipeline for YOLO
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            # Normalize to [0, 1] range - YOLO doesn't use ImageNet normalization
            transforms.Lambda(lambda x: x/255.0)
        ])


    def validate_image(self, image):
        """
        Validates image properties
        Returns: Boolean indicating if image is valid
        """

        if isinstance(image, np.ndarray):
            # Check if image is empty
            if image.size == 0:
                return False
            # Check if image has valid shape
            if len(image.shape) not in [2,3]:
                return False
            # Check if image has valid values
            if image.dtype == np.uint8:
                if image.max() > 255 or image.min() < 0:
                    return False
            return True
        elif isinstance(image, Image.Image):
            # Check if image is empty
            if image.size[0] == 0 or image.size[1] == 0:
                return False
            return True
        return False
    

    def load_image(self, image_input):
        """
        Loads image from various input formats
        Args:
            image_input: Can be:
                - Path to image (str or Path)
                - URL to image (str starting with http)
                - PIL Image
                - numpy array
                - bytes or BytesIO object
        Returns:
            tuple: (PIL Image, original_size)
        """
        original_image = None
        
        # Handle different input types
        if isinstance(image_input, (str, Path)):
            # Handle URLs
            if str(image_input).startswith(('http://', 'https://')):
                try:
                    with urllib.request.urlopen(image_input) as response:
                        image_data = response.read()
                    original_image = Image.open(io.BytesIO(image_data))
                except Exception as e:
                    raise ValueError(f"Error loading image from URL: {e}")
            else:
                # Handle local file paths
                try:
                    original_image = Image.open(image_input)
                except Exception as e:
                    raise ValueError(f"Error loading image from path: {e}")
                
        # Handle PIL Images
        elif isinstance(image_input, Image.Image):
            original_image = image_input

        # Handle numpy arrays
        elif isinstance(image_input, np.ndarray):
            if not self.validate_image(image_input):
                raise ValueError("Invalid numpy array format")
            
            # Convert grayscale to RGB if necessary
            if len(image_input.shape) == 2:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif len(image_input.shape) == 3 and image_input.shape[2] == 1:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif len(image_input.shape) == 3 and image_input.shape[2] == 4:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB)
            elif len(image_input.shape) == 3 and image_input.shape[2] == 3:
                image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            
            original_image = Image.fromarray(image_input)

        # Handle bytes or BytesIO
        elif isinstance(image_input, (bytes, io.BytesIO)):
            try:
                if isinstance(image_input, bytes):
                    image_input = io.BytesIO(image_input)
                original_image = Image.open(image_input)
            except Exception as e:
                raise ValueError(f"Error loading image from bytes: {e}")
        
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")
        
        # Convert to RGB if necessary
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        return original_image, original_image.size
    

    def preprocess(self, image_input):
        """
        Preprocesses image for YOLO model
        Args:
            image_input: URL/Numpy array/PIL Image/Path to local file/bytes
        Returns:
            tuple: (tensor_image, original_size)
            - tensor_image: Preprocessed image tensor of shape (1, 3, input_size, input_size)
            - original_size: Original image size (width, height)
        """
        # Load and validate image
        original_image, original_size = self.load_image(image_input)

        # Apply transformations
        tensor_image = self.transform(original_image)

        # Add batch dimension
        tensor_image = tensor_image.unsqueeze(0)

        return tensor_image, original_size
    

    




    



            

