import torch

class Config:
    # Model parameters
    IMAGE_SIZE = 448
    GRID_SIZE = 7
    NUM_BOXES = 2
    NUM_CLASSES = 20

    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.0005
    EPOCHS = 100

    # Data parameters
    TRAIN_DIR = 'data/processed/train'
    VAL_DIR = 'data/processed/val'
    WEIGHTS_DIR = 'data/weights'

    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prediction parameters
    CONF_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4

    # Class names (PASCAL VOC)
    CLASS_NAMES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
