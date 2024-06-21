from math import ceil
from PIL import Image, ImageDraw
import os
import copy
import xml.etree.ElementTree as ET

def resize_image_div(img, size=8):
    # Get original image dimensions
    height = img.size[0]
    width = img.size[1]
    # Calculate nearest multiples of 8 for each dimension
    new_height = int((ceil(height / size)) * size)
    new_width = int((ceil(width / size)) * size)
    
    return img.resize((new_height, new_width))


def parse_annotation(annotation_path, image_dir='/home/jovyan/ddorin/generative_augmentation/augmenter_pipeline/VOC2007/JPEGImages'):
    # Read XML file
    with open(annotation_path, 'r') as f:
        annotation_xml = f.read()
    # Parse XML
    root = ET.fromstring(annotation_xml)
    # Get filename
    filename = root.find('filename').text

    # Check if image file exists
    image_path = os.path.join(image_dir, filename)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f'Image file {image_path} not found')

    # Initialize dictionary to store objects
    objects = {}

    # Iterate over objects in annotation
    for i, obj in enumerate(root.findall('object')):
        # Get object name
        obj_name = obj.find('name').text
        
        # Get bounding box coordinates
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        # Create black PIL image with the same size as original image
        img_width = int(root.find('size/width').text)
        img_height = int(root.find('size/height').text)
        pil_img = Image.new('L', (img_width, img_height))

        # Draw white bounding box on the image
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([xmin, ymin, xmax, ymax], fill=255)

        # Add image to list of object images
        if obj_name not in objects:
            objects[obj_name] = []
        objects[obj_name].append((pil_img, i))

    return filename, objects


def modify_annotation(old_annotation_path, image_filename, new_object_name, selected_object_index):
    # Load the old annotation
    with open(old_annotation_path, 'r') as f:
        annotation_xml = f.read()
    root = ET.fromstring(annotation_xml)

    # Create a deep copy of the annotation
    new_root = copy.deepcopy(root)

    # Remove unwanted elements from the copy
    for elem in ['segmented', 'source', 'owner']:
        for e in new_root.findall(elem):
            new_root.remove(e)
            
    new_root.find('filename').text = image_filename + '_modified.jpg'
    new_root.find('folder').text = 'VOC2007_augmentation'
    
    # Update the name of the selected object in the copy
    objects = new_root.findall('object')
    for i, obj in enumerate(objects):
        if i == selected_object_index:
            obj.find('name').text = new_object_name
            
    new_annotation_path = '/home/jovyan/ddorin/generative_augmentation/augmenter_pipeline/VOC2007_augmentation/Annotations/' + image_filename + '_modified.xml'
    # Save the modified annotation
    tree = ET.ElementTree(new_root)
    tree.write(new_annotation_path)


