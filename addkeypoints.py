import os
import json
from PIL import Image
from xml.etree.ElementTree import Element, SubElement, ElementTree

# Paths
coco_annotation_file = 'path/to/instances_train2017.json'
coco_image_dir = 'path/to/COCO/images/'
output_image_dir = 'path/to/resized_images/'
output_annotation_dir = 'path/to/adjusted_annotations/'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_annotation_dir, exist_ok=True)

# Target size (e.g., Pascal VOC average dimensions)
target_width, target_height = 500, 375

# Load COCO annotations
with open(coco_annotation_file, 'r') as f:
    coco = json.load(f)

# Create a mapping for categories
categories = {cat['id']: cat['name'] for cat in coco['categories']}

def resize_image_and_boxes(image_info, annotations, target_width, target_height):
    """Resize an image and adjust bounding boxes and keypoints."""
    img_path = os.path.join(coco_image_dir, image_info['file_name'])
    with Image.open(img_path) as img:
        orig_width, orig_height = img.size
        
        # Resize image
        img_resized = img.resize((target_width, target_height), Image.ANTIALIAS)
        img_resized.save(os.path.join(output_image_dir, image_info['file_name']))
        
        # Scale factors
        scale_x = target_width / orig_width
        scale_y = target_height / orig_height
        
        # Adjust bounding boxes and keypoints
        adjusted_annotations = []
        for ann in annotations:
            bbox = ann['bbox']  # COCO format: [xmin, ymin, width, height]
            x_min = bbox[0] * scale_x
            y_min = bbox[1] * scale_y
            width = bbox[2] * scale_x
            height = bbox[3] * scale_y
            x_max = x_min + width
            y_max = y_min + height
            
            # Adjust keypoints
            keypoints = ann.get('keypoints', [])
            adjusted_keypoints = []
            for i in range(0, len(keypoints), 3):
                x, y, v = keypoints[i:i+3]
                if v > 0:  # Only scale valid keypoints
                    x = x * scale_x
                    y = y * scale_y
                adjusted_keypoints.append((x, y, v))
            
            # Add adjusted annotation
            adjusted_annotations.append({
                'category': categories[ann['category_id']],
                'bbox': [x_min, y_min, x_max, y_max],
                'keypoints': adjusted_keypoints
            })
        return adjusted_annotations

def save_as_voc_xml(image_info, adjusted_annotations):
    """Save annotations in Pascal VOC XML format, including keypoints."""
    xml_root = Element('annotation')
    folder = SubElement(xml_root, 'folder')
    folder.text = 'VOC_COCO'
    
    filename = SubElement(xml_root, 'filename')
    filename.text = image_info['file_name']
    
    size = SubElement(xml_root, 'size')
    SubElement(size, 'width').text = str(target_width)
    SubElement(size, 'height').text = str(target_height)
    SubElement(size, 'depth').text = '3'  # Assuming RGB images
    
    for ann in adjusted_annotations:
        # Object element for bounding boxes
        obj = SubElement(xml_root, 'object')
        name = SubElement(obj, 'name')
        name.text = ann['category']
        
        bndbox = SubElement(obj, 'bndbox')
        SubElement(bndbox, 'xmin').text = str(int(ann['bbox'][0]))
        SubElement(bndbox, 'ymin').text = str(int(ann['bbox'][1]))
        SubElement(bndbox, 'xmax').text = str(int(ann['bbox'][2]))
        SubElement(bndbox, 'ymax').text = str(int(ann['bbox'][3]))
        
        # Keypoints element
        keypoints = SubElement(obj, 'keypoints')
        for idx, (x, y, v) in enumerate(ann['keypoints']):
            kp = SubElement(keypoints, 'keypoint')
            kp_id = SubElement(kp, 'id')
            kp_id.text = str(idx)
            kp_x = SubElement(kp, 'x')
            kp_x.text = str(int(x)) if v > 0 else 'NaN'
            kp_y = SubElement(kp, 'y')
            kp_y.text = str(int(y)) if v > 0 else 'NaN'
            kp_v = SubElement(kp, 'visibility')
            kp_v.text = str(v)
    
    # Save XML file
    output_file = os.path.join(output_annotation_dir, f"{image_info['file_name'].split('.')[0]}.xml")
    tree = ElementTree(xml_root)
    tree.write(output_file)

# Process each image
for img_info in coco['images']:
    # Get annotations for the current image
    img_id = img_info['id']
    annotations = [ann for ann in coco['annotations'] if ann['image_id'] == img_id]
    
    # Resize image and adjust bounding boxes and keypoints
    adjusted_annotations = resize_image_and_boxes(img_info, annotations, target_width, target_height)
    
    # Save adjusted annotations in Pascal VOC XML format
    save_as_voc_xml(img_info, adjusted_annotations)

print("Image resizing, bounding box, and keypoint adjustment complete!")
