import os
import copy
import xml.etree.ElementTree as ET
from math import ceil
from PIL import Image, ImageDraw
from typing import Any, Dict, List, Optional, Tuple

def PILmask_from_bboxCoords(coords: Tuple[int, int, int, int], img_width: int, img_height: int) -> Image.Image:
    xmin, ymin, xmax, ymax = coords
    # Create black PIL mask
    pil_mask = Image.new('L', (img_width, img_height))
    # Draw white bounding box on the image
    draw = ImageDraw.Draw(pil_mask)
    draw.rectangle([xmin, ymin, xmax, ymax], fill=255)
    return pil_mask

def parse_annotation(annotation_path, images_dir='./VOC2007/JPEGImages'):
    with open(annotation_path, 'r') as f:
        annotation_xml = f.read()
    root = ET.fromstring(annotation_xml)
    filename = root.find('filename').text

    image_path = os.path.join(images_dir, filename)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f'Image file {image_path} not found')

    objects = {}
    for i, obj in enumerate(root.findall('object')):
        obj_name = obj.find('name').text
        
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        coords = (xmin, ymin, xmax, ymax)
        
        img_width = int(root.find('size/width').text)
        img_height = int(root.find('size/height').text)
        pil_mask = PILmask_from_bboxCoords(coords, img_width, img_height)

        if obj_name not in objects:
            objects[obj_name] = []
        objects[obj_name].append((pil_mask, i))

    return filename, objects

def modify_annotation(old_annotation_filename, new_annotation_filename, new_object_class, replaced_object_index, name_of_augmentation_set_dir):
    # Load the old annotation
    old_annotation_path = os.path.join('./VOC2007/Annotations', old_annotation_filename+'.xml')
    with open(old_annotation_path, 'r') as f:
        annotation_xml = f.read()
    root = ET.fromstring(annotation_xml)

    new_root = copy.deepcopy(root)

    for elem in ['segmented', 'source', 'owner']:
        for e in new_root.findall(elem):
            new_root.remove(e)

    new_root.find('filename').text = new_annotation_filename+'.jpg'
    new_root.find('folder').text = 'VOC2007_augmentation'

    objects = new_root.findall('object')
    for i, obj in enumerate(objects):
        if i == replaced_object_index:
            obj.find('name').text = new_object_class

            # Check for 'part' tags and remove them
            parts = obj.findall('part')
            for part in parts:
                obj.remove(part)

    new_annotation_path = os.path.join(f'./{name_of_augmentation_set_dir}/Annotations', new_annotation_filename+'.xml')
    # Save the modified annotation
    tree = ET.ElementTree(new_root)
    tree.write(new_annotation_path)


