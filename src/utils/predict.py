import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class YOLOPredictor:
    def __init__(self, model_path, S=7, B=2, C=20, conf_threshold=0.5, nms_threshold=0.4):
        """
        Initialize YOLO predictor
        Args:
            model_path: Path to saved model weights
            S: Grid size
            B: Number of boxes per grid cell
            C: Number of classes
            conf_threshold: Confidence threshold for detections
            nms_threshold: Non-maximum suppression threshold
        """
        self.S = S
        self.B = B
        self.C = C
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLOv1(S=S, B=B, C=C).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        
        # Class names (modify according to your dataset)
        self.class_names = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

    def preprocess_image(self, image_path):
        """
        Preprocess image for YOLO model
        """
        # Read and convert image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
        
        # Store original image size
        orig_size = image.size
        
        # Transform image
        image = self.transform(image)
        
        return image.unsqueeze(0).to(self.device), orig_size

    def decode_predictions(self, predictions, orig_size):
        """
        Decode YOLO predictions to bounding boxes
        """
        predictions = predictions.squeeze(0)
        bboxes = []
        class_labels = []
        confidences = []
        
        cell_size = 1.0 / self.S
        
        # For each cell in the grid
        for i in range(self.S):
            for j in range(self.S):
                # Get class probabilities
                class_probs = predictions[i, j, :self.C]
                
                # Get confidence scores for both boxes
                conf_scores = [
                    predictions[i, j, self.C],
                    predictions[i, j, self.C + 5]
                ]
                
                # Get box coordinates
                box1 = predictions[i, j, self.C+1:self.C+5]
                box2 = predictions[i, j, self.C+6:self.C+10]
                boxes = [box1, box2]
                
                # For each box
                for box_idx in range(self.B):
                    conf_score = conf_scores[box_idx]
                    
                    # Skip if confidence is below threshold
                    if conf_score < self.conf_threshold:
                        continue
                    
                    # Get box coordinates
                    box = boxes[box_idx]
                    x = (j + box[0]) * cell_size
                    y = (i + box[1]) * cell_size
                    w = box[2] * cell_size
                    h = box[3] * cell_size
                    
                    # Convert to corner coordinates
                    x1 = (x - w/2) * orig_size[0]
                    y1 = (y - h/2) * orig_size[1]
                    x2 = (x + w/2) * orig_size[0]
                    y2 = (y + h/2) * orig_size[1]
                    
                    # Get class label
                    class_idx = torch.argmax(class_probs)
                    class_prob = class_probs[class_idx]
                    
                    # Store prediction
                    bboxes.append([x1, y1, x2, y2])
                    class_labels.append(class_idx.item())
                    confidences.append((conf_score * class_prob).item())
        
        return np.array(bboxes), np.array(class_labels), np.array(confidences)

    def non_max_suppression(self, bboxes, class_labels, confidences):
        """
        Apply non-maximum suppression to remove overlapping boxes
        """
        if len(bboxes) == 0:
            return [], [], []
            
        # Convert to numpy arrays
        bboxes = np.array(bboxes)
        confidences = np.array(confidences)
        class_labels = np.array(class_labels)
        
        # Get indices of sorted confidences (highest first)
        indices = np.argsort(confidences)[::-1]
        
        keep = []
        while indices.size > 0:
            # Keep highest confidence detection
            current_idx = indices[0]
            keep.append(current_idx)
            
            if indices.size == 1:
                break
                
            # Get IOUs of all remaining boxes with the current box
            current_box = bboxes[current_idx]
            remaining_boxes = bboxes[indices[1:]]
            
            # Calculate IOUs
            xx1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            yy1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            xx2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            yy2 = np.minimum(current_box[3], remaining_boxes[:, 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            box1_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            box2_area = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
            union = box1_area + box2_area - intersection
            
            iou = intersection / union
            
            # Keep boxes with IOUs below threshold
            indices = indices[1:][iou < self.nms_threshold]
        
        return bboxes[keep], class_labels[keep], confidences[keep]

    def predict(self, image_path, return_annotated=True):
        """
        Predict objects in image
        Args:
            image_path: Path to image or numpy array
            return_annotated: Whether to return annotated image
        Returns:
            Tuple of (boxes, labels, scores) and optionally annotated image
        """
        # Preprocess image
        image_tensor, orig_size = self.preprocess_image(image_path)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Decode predictions
        bboxes, class_labels, confidences = self.decode_predictions(predictions, orig_size)
        
        # Apply non-maximum suppression
        bboxes, class_labels, confidences = self.non_max_suppression(
            bboxes, class_labels, confidences
        )
        
        if return_annotated:
            # Load original image for annotation
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path.copy()
            
            # Draw boxes and labels
            for box, label, conf in zip(bboxes, class_labels, confidences):
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, box)
                
                # Draw box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label_text = f"{self.class_names[label]}: {conf:.2f}"
                cv2.putText(image, label_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return (bboxes, class_labels, confidences), image
        
        return (bboxes, class_labels, confidences)

def plot_predictions(image, detections):
    """
    Plot image with detections
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# # Example usage
# def main():
#     # Initialize predictor
#     predictor = YOLOPredictor(
#         model_path='path/to/your/model.pth',
#         conf_threshold=0.5,
#         nms_threshold=0.4
#     )
    
#     # Make prediction
#     image_path = 'path/to/your/image.jpg'
#     detections, annotated_image = predictor.predict(image_path)
    
#     # Plot results
#     plot_predictions(annotated_image, detections)
    
#     # Print detections
#     bboxes, labels, scores = detections
#     for box, label, score in zip(bboxes, labels, scores):
#         print(f"Class: {predictor.class_names[label]}, Confidence: {score:.2f}")
#         print(f"Box: {box}")

# if __name__ == "__main__":
#     main()