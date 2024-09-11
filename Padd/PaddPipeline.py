import numpy as np
import torch
import random
import copy
import cv2
from PIL import Image, ImageOps
from typing import Tuple, List
from .models import PowerPaintControlNet

class ObjectAdder():
    def __init__(self, 
                 device: str = "cuda") -> None:
        """
        Initializes the model which add object to scene.

        Args:
        device (str): Describes the device on which the model will run. Defaults to "cuda".
        """
        self.device = device
        self.weight_dtype = torch.float16

        self.PowerPaint = PowerPaintControlNet(device=self.device)
        self.depth_estimator = self.PowerPaint.depth_estimator
        self.feature_extractor = self.PowerPaint.feature_extractor

    def get_depth_map(self, image: Image.Image) -> Image.Image:
        """
        Gets the depth map of the given image.

        Args:
        image (Image.Image): The input image.

        Returns:
        Image.Image: The depth map of the input image.
        """
        size = image.size
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image.resize(size)

    def resize_and_random_flip(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Resizes and randomly flips the given image and mask.

        Args:
        image (Image.Image): The input image.
        mask (Image.Image): The input mask.

        Returns:
        Tuple[Image.Image, Image.Image]: The resized and flipped image and mask.
        """
        # Randomly select a resizing factor between 0.5 and 1.5
        factor = random.uniform(0.2, 0.5)
        # Calculate the new size
        new_size = (int(image.width * factor), int(image.height * factor))
        # Resize the images
        resized_image = image.resize(new_size)
        resized_mask = mask.resize(new_size)
        # Randomly flip the images
        if random.choice([True, False]):
            resized_image = ImageOps.mirror(resized_image)
            resized_mask = ImageOps.mirror(resized_mask)

        return resized_image, resized_mask
    
    def calculate_average_depth_and_bottom_point(self, depth_map: Image.Image, mask: Image.Image) -> Tuple[float, np.ndarray]:
        """
        Calculates the average depth and bottom point of the given depth map and mask.

        Args:
        depth_map (Image.Image): The input depth map.
        mask (Image.Image): The input mask.

        Returns:
        Tuple[float, np.ndarray]: The average depth and bottom point.
        """
        # Ensure both inputs are numpy arrays
        depth_map = np.array(depth_map)
        mask = np.array(mask)
        # Ensure mask is binary
        mask = (mask > 0).astype(np.uint8)
        if len(depth_map.shape) == 3:
            depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
        # Apply mask to depth map
        masked_depth = depth_map * mask
        # Calculate average depth of non-zero (masked) pixels
        total_depth = np.sum(masked_depth)
        num_pixels = np.sum(mask)
        
        if num_pixels > 0:
            average_depth = total_depth / num_pixels
        else:
            average_depth = 0
        # Find the bottom point (lowest non-zero pixel in the mask)
        bottom_point = None
        non_zero_indices = np.argwhere(mask)
        if non_zero_indices.size > 0:
            bottom_point = non_zero_indices[-1]
        
        return average_depth, bottom_point
    
    def sample_random_coordinates(self, array: np.ndarray, num_samples: int = 1) -> List[Tuple[int, int]]:
        """
        Samples random coordinates from the given array.

        Args:
        array (np.ndarray): The input array.
        num_samples (int): The number of samples to take. Defaults to 1.

        Returns:
        List[Tuple[int, int]]: The sampled coordinates.
        """
        # Find indices where the value is 1
        indices = np.argwhere(array == 1)
        # If num_samples is greater than available indices, adjust the sample size
        num_samples = min(num_samples, len(indices))
        # Randomly sample the indices
        sampled_indices = indices[np.random.choice(len(indices), num_samples, replace=False)]
        # Return the sampled coordinates as a list of tuples
        return [tuple(coord) for coord in sampled_indices]


    def blend_condition_images(self, scene: Image.Image, image: Image.Image, mask_image: Image.Image, position: Tuple[int, int]) -> Image.Image:
        """
        Blends the condition images.

        Args:
        scene (Image.Image): The scene image.
        image (Image.Image): The image to blend.
        mask_image (Image.Image): The mask image.
        position (Tuple[int, int]): The position to blend at.

        Returns:
        Image.Image: The blended image.
        """
        scene = copy.copy(scene)
        # Paste the small image onto the large image at the defined position
        scene.paste(image, position, mask_image)
        return scene


    def __call__(self,
                scene_image: Image.Image,
                object_image: Image.Image,
                object_image_mask: Image.Image,
                prompt: str,
                seed: int) -> Tuple[Image.Image, Image.Image]:
        """
        Calls the model to generate an image.

        Args:
        scene_image (Image.Image): The scene image.
        object_image (Image.Image): The object image.
        object_image_mask (Image.Image): The object image mask.
        prompt (str): The prompt to use.
        seed (int): The seed to use.

        Returns:
        Tuple[Image.Image, Image.Image]: The generated image and the control image.
        """
        np.random.seed(seed)
        random.seed(seed)
        object_depth = self.get_depth_map(object_image)
        scene_depth = self.get_depth_map(scene_image)
        
        object_image_mask = object_image_mask.convert('L')
        size = (object_depth.size[0] // 8, object_depth.size[0] // 8)
        object_depth.resize(size)
        object_image_mask.resize(size)
        
        resized_object_depth, resized_object_mask = self.resize_and_random_flip(object_depth, object_image_mask)
        average_depth, bottom_point = self.calculate_average_depth_and_bottom_point(resized_object_depth, resized_object_mask)

        location_array = (np.array(scene_depth) > 195).astype(float)
        sampled_coords = self.sample_random_coordinates(location_array, num_samples=1)

        object_point = bottom_point
        scene_point = np.array(sampled_coords[0])[:2]
        paste_x = (scene_point[0] - object_point[0])
        paste_y = (scene_point[1] - object_point[1])
        position = (paste_y, paste_x)

        controlnet_image = self.blend_condition_images(scene_depth, resized_object_depth, resized_object_mask, position)
        mask = Image.new("L", scene_image.size, 0)
        # Paste the small image onto the large image at the defined position
        mask.paste(resized_object_mask, position)

        img = {"image": scene_image, "mask": mask}

        new_image, controlnet_image = self.PowerPaint(
                input_image=img,
                control_type="depth",
                prompt=prompt,
                ddim_steps=50,
                scale=5,
                seed=seed,
                controlnet_conditioning_scale=0.8,
                input_control_image=controlnet_image)

        return new_image, controlnet_image

