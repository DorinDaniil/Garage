import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import copy
import cv2
from PIL import Image, ImageOps
from typing import Tuple, List
from .models import PowerPaintControlNet
from transformers import AutoProcessor, AutoModelForCausalLM

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

        print('Florence model...')
        self.florence_model_id = 'microsoft/Florence-2-large'
        self.florence_model = AutoModelForCausalLM.from_pretrained(self.florence_model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
        self.florence_processor = AutoProcessor.from_pretrained(self.florence_model_id, trust_remote_code=True)

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

    def resize_and_random_flip(self, image, mask, position_bbox = None):
        # Randomly select a resizing factor between 0.5 and 1.5
        factor = random.uniform(0.2, 0.5)
        # Calculate the new size
        if position_bbox is not None:
            x1, y1, x2, y2 = position_bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            factor = np.sqrt( (x2 - x1) * (y2 - y1)  / image.width / image.height )
            print(image.width*factor ,image.height*factor)
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
    
    def sample_random_coordinates(self, array: np.ndarray, num_samples: int = 1, without_overlaps: bool = True, surface_size: Tuple = () , image_sizes : List = []) -> List[Tuple[int, int]]:
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
        
        
        if without_overlaps:
            sampled_indices = []
            mask = np.ones_like(array)
            for i in range(num_samples):
                # print(array.shape, indices.shape)
                img_size = image_sizes[i]
                ind = indices[np.random.choice(len(indices),1, replace=False)][0]
                sampled_indices.append(ind)
                # print(ind)
                # print(ind[0],ind[0] + int(img_size[0]) , ind[1], ind[1]+int(img_size[1]))
                mask[ind[0]:ind[0] + int(img_size[0])][ind[1]:ind[1]+int(img_size[1])] = 0
                indices = np.argwhere( array*mask == 1)
        else:
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

    def ground_by_florence(self, image , task_prompt, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.florence_processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
        generated_ids = self.florence_model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
        )
        generated_text = self.florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.florence_processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        return parsed_answer  
    def preprocess_scene_image(self, scene: Image):
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        text_input = "A small surface at the bottom, or some position"
        results = self.ground_by_florence(scene ,task_prompt, text_input=text_input)

        # Just search a big surface, but not large than 0.6 of volume scene
        # Also caption is not checked, we use only grounded bboxes 
          
        result_bboxes = results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
        image_volume = scene.size[0]*scene.size[1]
        volumes = [ (bbox[2] - bbox[0])*(bbox[3]-bbox[1]) /image_volume for bbox in result_bboxes ]
        filtered_volumes = [ vol if vol < 0.6 else 0 for vol in volumes ]
        if len(filtered_volumes) == 0 : # Then use lower half of scene
            filtered_volumes = [[0,0,scene.size[0], int(scene.size[1]/2)]]     
        best_bbox_index = np.argmax(filtered_volumes)
        position_bbox = result_bboxes[best_bbox_index]
        return position_bbox

    def filter_location(self, position_bbox, scene_depth):
        q = np.quantile(scene_depth,q = 0.5 )
        location_array = (np.array(scene_depth) > q).astype(float)

        if position_bbox is not None:
            x1, y1, x2, y2 = position_bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            new_location_array = np.zeros_like(location_array)
            new_location_array[y1:y2+1, x1:x2+1] = location_array[y1:y2+1, x1:x2+1]
            location_array = new_location_array
        return location_array
    
    def generate_new_object(self, object_image, object_image_mask, scene_image, scene_depth, position_bbox, sampled_coord, prompt, seed=None):
        object_depth = self.get_depth_map(object_image)
        
        object_image_mask = object_image_mask.convert('L')
        size = (object_depth.size[0] // 8, object_depth.size[0] // 8)
        object_depth.resize(size)
        object_image_mask.resize(size)

        resized_object_depth, resized_object_mask = self.resize_and_random_flip(object_depth, object_image_mask, position_bbox=position_bbox)
        average_depth, bottom_point = self.calculate_average_depth_and_bottom_point(resized_object_depth, resized_object_mask)
        
    
        object_point = bottom_point
        scene_point = np.array(sampled_coord)[:2]
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
        return new_image , controlnet_image
    
    def __call__(self,
                 scene_image,
                 object_images,
                 object_image_masks,
                 prompts,
                 seed,
                 ):
        
        np.random.seed(seed)
        random.seed(seed)

        scene_depth = self.get_depth_map(scene_image)

        new_images = []
        controlnet_images = []

        if type(object_images) == type([]):
            # search position bbox on the scene
            position_bbox = self.preprocess_scene_image(scene_image)
            # plt.imshow(scene_depth)
            location_array = self.filter_location(position_bbox , scene_depth)
            # print(position_bbox)
            # print(np.sum(location_array), location_array.shape)
            surface_size = (position_bbox[2] - position_bbox[0] , position_bbox[3] - position_bbox[1] )
            image_sizes = [img.size for img in object_images]
            sampled_coords = self.sample_random_coordinates(location_array, num_samples=len(object_images), without_overlaps=True, surface_size = surface_size , image_sizes=image_sizes)
            # print(sampled_coords)
            new_image = scene_image
            for i in range(len(object_images)):
                object_image, object_image_mask, sampled_coord, prompt = object_images[i], object_image_masks[i], sampled_coords[i], prompts[i]

                new_image , controlnet_image = self.generate_new_object(
                        object_image = object_image, 
                        object_image_mask = object_image_mask, 
                        scene_image = new_image, 
                        scene_depth = scene_depth, 
                        position_bbox = position_bbox, 
                        sampled_coord = sampled_coord, 
                        prompt = prompt, 
                        seed=seed
                    )
                
                new_images.append(new_image)
                controlnet_images.append(controlnet_image)

        return new_images, controlnet_images

