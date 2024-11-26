import json
import random
from pycocotools.coco import COCO

# File paths
annotation_file = 'path/to/instances_train2017.json'
output_dir = '/coco_output/'

# Load COCO dataset
coco = COCO(annotation_file)

# Get the category ID for 'person'
person_category_id = coco.getCatIds(catNms=['person'])[0]

# Get all annotations for 'person'
person_ann_ids = coco.getAnnIds(catIds=[person_category_id])
person_annotations = coco.loadAnns(person_ann_ids)

# Group annotations by image
image_to_objects = {}
for ann in person_annotations:
    img_id = ann['image_id']
    if img_id not in image_to_objects:
        image_to_objects[img_id] = []
    image_to_objects[img_id].append(ann)

# Filter images with at least one 'person' object
person_images = list(image_to_objects.keys())

# Sample images to match Pascal VOC numbers
random.seed(42)  # For reproducibility
sampled_images = random.sample(person_images, 2008)  # 2008 images

# Ensure 4690 objects are sampled
sampled_objects = []
for img_id in sampled_images:
    sampled_objects.extend(image_to_objects[img_id])
    if len(sampled_objects) >= 4690:
        break

# Truncate to exactly 4690 objects if necessary
sampled_objects = sampled_objects[:4690]

# Save the list of sampled images and objects for reference
sampled_image_ids = set([obj['image_id'] for obj in sampled_objects])
sampled_image_files = [coco.loadImgs(img_id)[0] for img_id in sampled_image_ids]

print(f"Sampled {len(sampled_image_files)} images and {len(sampled_objects)} person objects.")

# Optionally, download the images
import requests
import os

output_image_dir = os.path.join(output_dir, 'images/')
os.makedirs(output_image_dir, exist_ok=True)

for img_info in sampled_image_files:
    img_url = img_info['coco_url']
    img_data = requests.get(img_url).content
    img_path = os.path.join(output_image_dir, img_info['file_name'])
    with open(img_path, 'wb') as f:
        f.write(img_data)

print("Downloaded sampled images.")
