import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image, ImageOps, ImageDraw, ImageEnhance
import numpy as np
import random
import cv2
import os
import copy
from typing import Optional, Tuple, List
from .models import PowerPaintControlNet
from .models import PhysicsModel

class ObjectAdder:
    def __init__(self, device: str = "cuda", physic_model='v0.1') -> None:
        """
        Initializes the model which adds an object to the scene.

        Args:
            device (str): The device on which the model will run. Defaults to "cuda".
        """
        self.device = device
        self.weight_dtype = torch.float16

        self.PowerPaint = PowerPaintControlNet(device=self.device)
        self.depth_estimator = self.PowerPaint.depth_estimator
        self.feature_extractor = self.PowerPaint.feature_extractor

        print('Florence model...')
        self.florence_model_id = 'microsoft/Florence-2-large'
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            self.florence_model_id, 
            trust_remote_code=True, 
            torch_dtype='auto'
        ).eval().to(self.device)
        self.florence_processor = AutoProcessor.from_pretrained(self.florence_model_id, trust_remote_code=True)
        
        self.physic_model = physic_model
        if physic_model:
            script_directory = os.path.dirname(os.path.abspath(__file__))
            self.checkpoints_path = os.path.join(script_directory, "PhysicModel", "checkpoints", physic_model)
            self.physic_model = PhysicsModel.from_pretrain('/home/jovyan/afilatov/Augmentations/physics_model/ckpt/epoch_99/model.safetensors')


    def get_depth_map(self, image: Image.Image) -> Image.Image:
        """
        Gets the depth map of the given image.

        Args:
            image (Image.Image): The input image.

        Returns:
            Image.Image: The depth map of the input image.
        """
        size = image.size
        image_tensor = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad(), torch.autocast(self.device):
            depth_map = self.depth_estimator(image_tensor).predicted_depth

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
        return Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8)).resize(size)

    def resize_and_random_flip(self, image: Image.Image, 
                               mask: Image.Image, 
                               addition_resize_factor: float, 
                               position_bbox: Optional[Tuple[int, int, int, int]] = None
                               ) -> Tuple[Image.Image, Image.Image]:
        """
        Resizes and randomly flips the image and mask.

        Args:
            image (Image.Image): The input image.
            mask (Image.Image): The input mask.
            addition_resize_factor (float): Addition resize factor based on depth
            position_bbox (Optional[Tuple[int, int, int, int]]): The bounding box for resizing. Defaults to None.

        Returns:
            Tuple[Image.Image, Image.Image]: The resized and possibly flipped image and mask.
        """
        if position_bbox is not None:
            x1, y1, x2, y2 = position_bbox
            factor = np.sqrt((x2 - x1) * (y2 - y1) / image.width / image.height)
        else:
            factor = random.uniform(0.5, 1.5)

        new_size = (int(image.width * factor * addition_resize_factor), int(image.height * factor * addition_resize_factor))
        resized_image = image.resize(new_size)
        resized_mask = mask.resize(new_size)

        if random.choice([True, False]):
            resized_image = ImageOps.mirror(resized_image)
            resized_mask = ImageOps.mirror(resized_mask)
        return resized_image, resized_mask

    def calculate_average_depth_and_bottom_point(self, object_depth: Image.Image, object_mask: Image.Image) -> Tuple[float, Optional[np.ndarray]]:
        """
        Calculates the average depth and bottom point of the given object depth and object mask.

        Args:
            object_depth (Image.Image): The input object depth.
            object_mask (Image.Image): The input mask.

        Returns:
            Tuple[float, Optional[np.ndarray]]: The average depth and bottom point.
        """
        object_depth = np.array(object_depth)
        object_mask = np.array(object_mask)
        object_mask = (object_mask > 0).astype(np.uint8)
        if len(object_depth.shape) == 3:
            object_depth = cv2.cvtColor(object_depth, cv2.COLOR_BGR2GRAY)
        masked_depth = object_depth * object_mask
        total_depth = np.sum(masked_depth)
        num_pixels = np.sum(object_mask)
        
        if num_pixels > 0:
            average_depth = total_depth / num_pixels
        else:
            average_depth = 0
        bottom_point = None
        non_zero_indices = np.argwhere(object_mask)
        if non_zero_indices.size > 0:
            bottom_point = non_zero_indices[-1]
        
        return average_depth, bottom_point
    
    def sample_random_coordinates(
        self, 
        array: np.ndarray, 
        num_samples: int = 1, 
        without_overlaps: bool = True, 
        surface_size: Optional[Tuple[int, int]] = None, 
        image_sizes: List[Tuple[int, int]] = []
    ) -> List[Tuple[int, int]]:
        """
        Samples random coordinates from the given binary array.

        Args:
            array (np.ndarray): The input binary array.
            num_samples (int): The number of samples to take. Defaults to 1.
            without_overlaps (bool): If True, samples without overlaps. Defaults to True.
            surface_size (Optional[Tuple[int, int]]): Size of the surface to sample from. Not used in the current implementation.
            image_sizes (List[Tuple[int, int]]): The sizes of the images corresponding to each sample.

        Returns:
            List[Tuple[int, int]]: The sampled coordinates as a list of (x, y) tuples.
        """
        indices = np.argwhere(array == 1)
        num_samples = min(num_samples, len(indices))
        without_overlaps = False
        if without_overlaps:
            sampled_indices = []
            mask = np.ones_like(array)
            for i in range(num_samples):
                img_size = image_sizes[i]
                ind = indices[np.random.choice(len(indices),1, replace=False)][0]
                sampled_indices.append(ind)
                mask[ind[0]:ind[0] + int(img_size[0])][ind[1]:ind[1]+int(img_size[1])] = 0
                indices = np.argwhere( array*mask == 1)
        else:
            sampled_indices = indices[np.random.choice(len(indices), num_samples, replace=False)]
        # Return the sampled coordinates as a list of tuples
        return [tuple(coord) for coord in sampled_indices]
    
    def blend_condition_images(self, scene_depth: Image.Image, 
                               object_depth: Image.Image, 
                               object_mask: Image.Image, 
                               object_mean_depth: float,
                               position_depth: float,
                               position: Tuple[int, int]) -> Image.Image:
        """
        Blends the condition images onto the scene.

        Args:
            scene_depth (Image.Image): The scene image depth.
            object_depth (Image.Image): The object depth image.
            object_mask (Image.Image): The mask image for blending.
            object_mean_depth (float): The object mean depth.
            position_depth (float): The sampled point on the scene depth.
            position (Tuple[int, int]): The position to blend the image at.

        Returns:
            Image.Image: The blended image.
        """
        brightness_ratio = position_depth / object_mean_depth

        enhancer = ImageEnhance.Brightness(object_depth)
        object_depth_adjusted = enhancer.enhance(brightness_ratio)

        scene_depth_pasted = scene_depth.copy()
        scene_depth_pasted.paste(object_depth_adjusted, position, object_mask)
        return scene_depth_pasted

    def ground_by_florence(
        self, 
        image: Image.Image, 
        task_prompt: str, 
        text_input: Optional[str] = None
    ) -> dict:
        """
        Grounds the provided image using the Florence model and returns the result.

        Args:
            image (Image.Image): The image to ground.
            task_prompt (str): The prompt for the grounding task.
            text_input (Optional[str]): Additional text input to enhance the prompt.

        Returns:
            dict: The parsed answer from the Florence model.
        """
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.florence_processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.weight_dtype)
        generated_ids = self.florence_model.generate(input_ids=inputs["input_ids"].cuda(),
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
     
    def preprocess_scene_image(self, scene: Image.Image) -> Tuple[float, float, float, float]:
        """
        Preprocesses the scene image and retrieves the best bounding box for placement.

        Args:
            scene (Image.Image): The scene image to preprocess.

        Returns:
            Tuple[float, float, float, float]: The coordinates of the best bounding box for placement.
        """
        task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
        text_input = "A small surface at the bottom, or some position"
        results = self.ground_by_florence(scene, task_prompt, text_input=text_input)

        # Just search a big surface, but not large than 0.6 of volume scene
        # Also caption is not checked, we use only grounded bboxes 
          
        result_bboxes = results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
        image_volume = scene.size[0]*scene.size[1]
        volumes = [(bbox[2] - bbox[0])*(bbox[3]-bbox[1]) / image_volume for bbox in result_bboxes ]
        filtered_volumes = [ vol if vol < 0.4 else 0 for vol in volumes ]
        if len(filtered_volumes) == 0 : # Then use lower half of scene
            filtered_volumes = [[0,0,scene.size[0], int(scene.size[1]/2)]]     
        best_bbox_index = np.argmax(filtered_volumes)
        position_bbox = result_bboxes[best_bbox_index]
        return position_bbox

    def filter_location(self, position_bbox: Tuple[int, int, int, int], scene_depth: np.ndarray) -> np.ndarray:
        """
        Filters the location based on the scene's depth information.

        Args:
            position_bbox (Tuple[int, int, int, int]): The bounding box of the position.
            scene_depth (np.ndarray): The depth data of the scene.

        Returns:
            np.ndarray: A filtered location array.
        """
        q = np.quantile(scene_depth,q = 0.5 )
        location_array = (np.array(scene_depth) > q).astype(float)

        if position_bbox is not None:
            x1, y1, x2, y2 = position_bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            new_location_array = np.zeros_like(location_array)
            new_location_array[y1:y2+1, x1:x2+1] = location_array[y1:y2+1, x1:x2+1]
            location_array = new_location_array
        return location_array
    
    def crop_mask_and_image(
        self, 
        mask_image: Image.Image, 
        original_image: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Crops the mask and the original image based on the bounding box of the mask.

        Args:
            mask_image (Image.Image): The mask image.
            original_image (Image.Image): The original image.

        Returns:
            Tuple[Image.Image, Image.Image]: A tuple containing the cropped mask and the cropped original image.
        """
        mask_array = np.array(mask_image)

        coords = np.argwhere(mask_array)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        cropped_mask = mask_image.crop((x_min, y_min, x_max + 1, y_max + 1))
        cropped_image = original_image.crop((x_min, y_min, x_max + 1, y_max + 1))
        return cropped_mask, cropped_image
    
    def draw_bbox(
        self, 
        image: Image.Image, 
        coords: Tuple[int, int, int, int],
        outline: str = 'red', 
        width: int = 2
    ) -> Image.Image:
        """
        Draws a bounding box on the given image.

        Args:
            image (Image.Image): The image to draw on.
            coords (Tuple[int, int, int, int]): The coordinates of the bounding box (x1, y1, x2, y2).
            outline (str): The color of the bounding box outline. Defaults to 'red'.
            width (int): The width of the bounding box outline. Defaults to 2.

        Returns:
            Image.Image: The image with the bounding box drawn on it.
        """
        x1, y1, x2, y2 = coords
        draw = ImageDraw.Draw(image)
        draw.rectangle([(x1, y1), (x2, y2)], outline=outline, width=width)
        return image
    
    def generate_point_with_physic_models(
        self,
        object_image: Image.Image, 
        scene_image: Image.Image, 
        scene_depth_size: Tuple[int, int], 
        num_samples: int = 10, 
    ):
        rand_points = np.random.random((num_samples, 2))
        for point in rand_points:
            result_for_point = self.physic_model.run_inference(np.asarray(scene_image), np.asarray(object_image), point)
            if result_for_point:
                return [int(point[0]*scene_depth_size[0]), int(point[1]*scene_depth_size[1])]
        
        return None
    
    def generate_new_object(
        self, 
        object_image: Image.Image, 
        object_mask: Image.Image, 
        scene_image: Image.Image, 
        position_bbox: Tuple[int, int, int, int], 
        sampled_coord: Tuple[int, int], 
        prompt: str, 
        seed: int = 1
    ) -> Image.Image:
        """
        Generates a new object by blending the object image into the scene at the specified coordinates.

        Args:
            object_image (Image.Image): The image of the object to be added.
            object_mask (Image.Image): The mask image of the object.
            scene_image (Image.Image): The scene image where the object will be added.
            position_bbox (Tuple[int, int, int, int]): The bounding box for the object's position.
            sampled_coord (Tuple[int, int]): The coordinates to place the object.
            prompt (str): The prompt for generating the object.
            seed (int): A seed for random number generation. Defaults to 1.

        Returns:
            Image.Image: The scene image with the new object blended into it.
        """
        scene_depth = self.get_depth_map(scene_image)

        object_mask, object_image = self.crop_mask_and_image(object_mask, object_image)
        object_depth = self.get_depth_map(object_image)

        object_mask = object_mask.convert('L')

        sampled_point_depth = np.array(scene_depth)[int(sampled_coord[0]), int(sampled_coord[1])][0]
        addition_resize_factor = sampled_point_depth / 255

        resized_object_depth, resized_object_mask = self.resize_and_random_flip(object_depth, object_mask, addition_resize_factor, position_bbox=position_bbox)
        mean_object_depth, bottom_point = self.calculate_average_depth_and_bottom_point(resized_object_depth, resized_object_mask)

        object_point = bottom_point
        scene_point = np.array(sampled_coord)[:2]
        paste_x = (scene_point[0] - object_point[0])
        paste_y = (scene_point[1] - object_point[1])
        position = (paste_y, paste_x)

        controlnet_image = self.blend_condition_images(scene_depth, 
                                                       resized_object_depth, 
                                                       resized_object_mask,
                                                       mean_object_depth,
                                                       sampled_point_depth,
                                                       position)
        mask = Image.new("L", scene_image.size, 0)
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
    
    def __call__(
        self,
        scene_image: Image.Image,
        object_images: List[Image.Image],
        object_masks: List[Image.Image],
        prompts: List[str],
        seed: int
    ) -> Tuple[List[Image.Image], List[Image.Image]]:
        """
        Processes the scene image and adds the provided objects with their masks.

        Args:
            scene_image (Image.Image): The scene image where objects will be added.
            object_images (List[Image.Image]): A list of object images to be added to the scene.
            object_masks (List[Image.Image]): A list of masks corresponding to the object images.
            prompts (List[str]): A list of prompts for generating each object.
            seed (int): A seed for random number generation to ensure reproducibility.

        Returns:
            Tuple[List[Image.Image], List[Image.Image]]: A tuple containing:
                - A list of new images with objects added to the scene.
                - A list of control net images corresponding to the blended images.
        """
        
        np.random.seed(seed)
        random.seed(seed)

        new_images = []
        controlnet_images = []

        # Preprocessing
        object_masks = [object_mask.convert('L') for object_mask in object_masks]

        scene_depth = self.get_depth_map(scene_image)
        # search position bbox on the scene
        position_bbox = self.preprocess_scene_image(scene_image)
        location_array = self.filter_location(position_bbox , scene_depth)
        surface_size = (position_bbox[2] - position_bbox[0] , position_bbox[3] - position_bbox[1])
        image_sizes = [img.size for img in object_images]
        sampled_coords = self.sample_random_coordinates(location_array, 
                                                        num_samples=len(object_images), 
                                                        without_overlaps=True, 
                                                        surface_size = surface_size, 
                                                        image_sizes=image_sizes)
        
        sorted_sampled_coords = sorted(sampled_coords, key=lambda x: x[0])
        scene = scene_image.copy()
        for object_image, object_mask, prompt, sampled_coord in zip(object_images, object_masks, prompts, sorted_sampled_coords):
            if self.physic_model:
                physic_model_point = self.generate_point_with_physic_models(scene, object_image, np.asarray(scene_depth).shape)
                if physic_model_point is not None:
                    sampled_coord = physic_model_point
            scene, controlnet_image = self.generate_new_object(object_image, 
                                                               object_mask, 
                                                               scene,
                                                               position_bbox, 
                                                               sampled_coord, 
                                                               prompt, 
                                                               seed=seed
                                                               )
            
            new_images.append(scene)
            controlnet_images.append(controlnet_image)
        return new_images, controlnet_images
