from PIL import Image
import torch
import numpy as np
import os
import json
import random
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Dict, Any


def CombineImagesHorizontally(*images):
    if not images:
        raise ValueError("No images provided")

    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    combined_image = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return combined_image
