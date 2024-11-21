class YOLOLoss(nn.Module):
    """
    YOLO Loss Function
    """
    def __init__(self, S=7, B=2, C=20, coord_scale=5, noobj_scale=0.5):
        super(YOLOLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.coord_scale = coord_scale
        self.noobj_scale = noobj_scale
        
    def compute_iou(self, box1, box2):
        """Calculate IoU between box1 and box2"""
        # Convert predictions to corners
        box1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        box1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        box1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        box1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2
        box2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        box2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        box2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        box2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2
        
        # Intersection area
        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)
        intersection = torch.clamp((x2 - x1), 0) * torch.clamp((y2 - y1), 0)
        
        # Union area
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union = box1_area + box2_area - intersection
        
        return intersection / (union + 1e-6)
    
    def forward(self, predictions, targets):
        """
        predictions: (batch_size, S, S, C + B * 5)
        targets: (batch_size, S, S, C + 5)  # 5 for one ground truth box
        """
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        # Calculate IoU for the two predicted boxes
        iou_b1 = self.compute_iou(predictions[..., self.C+1:self.C+5], 
                                 targets[..., self.C+1:self.C+5])
        iou_b2 = self.compute_iou(predictions[..., self.C+6:self.C+10], 
                                 targets[..., self.C+1:self.C+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        
        # Get the box with highest IoU
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = targets[..., self.C].unsqueeze(3)  # Identity of object i
        
        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., self.C+6:self.C+10]
                + (1 - bestbox) * predictions[..., self.C+1:self.C+5]
            )
        )
        
        box_targets = exists_box * targets[..., self.C+1:self.C+5]
        
        # Take sqrt of width, height
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        # Calculate coordinate loss
        box_loss = self.coord_scale * torch.mean(
            exists_box * torch.sum((box_predictions - box_targets) ** 2, dim=-1)
        )
        
        # ==================== #
        #   FOR OBJECT LOSS   #
        # ==================== #
        pred_box = (
            bestbox * predictions[..., self.C+5:self.C+6]
            + (1 - bestbox) * predictions[..., self.C:self.C+1]
        )
        
        object_loss = torch.mean(
            exists_box * torch.sum((pred_box - exists_box) ** 2, dim=-1)
        )
        
        # ======================= #
        #   FOR NO OBJECT LOSS   #
        # ======================= #
        no_object_loss = self.noobj_scale * torch.mean(
            (1 - exists_box) 
            * torch.sum(
                (predictions[..., self.C:self.C+1] ** 2 
                 + predictions[..., self.C+5:self.C+6] ** 2),
                dim=-1
            )
        )
        
        # ================== #
        #   FOR CLASS LOSS  #
        # ================== #
        class_loss = torch.mean(
            exists_box 
            * torch.sum((predictions[..., :self.C] - targets[..., :self.C]) ** 2, dim=-1)
        )
        
        # Compute final loss
        total_loss = box_loss + object_loss + no_object_loss + class_loss
        return total_loss