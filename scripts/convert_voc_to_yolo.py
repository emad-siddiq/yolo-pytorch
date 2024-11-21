import os
import xml.etree.ElementTree as ET

# VOC Classes - match the order in config.py
CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def convert_annotation(xml_file, output_dir):
    """
    Convert VOC XML annotation to YOLO txt format
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image size
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    
    # Prepare output text file
    base_filename = os.path.splitext(os.path.basename(xml_file))[0]
    txt_file = os.path.join(output_dir, base_filename + '.txt')
    
    with open(txt_file, 'w') as out_file:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in CLASSES:
                continue
            
            class_id = CLASSES.index(class_name)
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (center_x, center_y, width, height)
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            
            # Write to output file
            out_file.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

def main():
    voc_dir = 'data/raw/VOCdevkit/VOC2012'
    train_output = 'data/processed/train'
    val_output = 'data/processed/val'
    
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)
    
    # Process train and val sets
    for set_type in ['train', 'val']:
        with open(os.path.join(voc_dir, f'ImageSets/Main/{set_type}.txt'), 'r') as f:
            image_ids = f.read().strip().split()
        
        output_dir = train_output if set_type == 'train' else val_output
        
        for image_id in image_ids:
            xml_file = os.path.join(voc_dir, f'Annotations/{image_id}.xml')
            convert_annotation(xml_file, output_dir)
            
            # Copy corresponding image
            src_image = os.path.join(voc_dir, f'JPEGImages/{image_id}.jpg')
            dst_image = os.path.join(output_dir, f'{image_id}.jpg')
            os.system(f'cp {src_image} {dst_image}')

if __name__ == '__main__':
    main()