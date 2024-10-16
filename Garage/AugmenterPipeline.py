import torch
import numpy as np
import random
from PIL import Image
from typing import List, Optional, Tuple
from accelerate import Accelerator

from .models import LLaMAModel
from .models import LLaVAModel
from .models import PowerPaintModel

class Augmenter:
    """
    A class that performs image augmentation by replacing objects in images.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initializes the Augmenter class.

        Args:
        device (str): The device to use for computations. Defaults to "cuda".
        """
        self.device = device
        self.accelerator = Accelerator()

        self._models = {
            "LLaMA": None,
            "LLaVA": LLaVAModel(device=self.device),
            "PowerPaint": PowerPaintModel(device=self.device)
        }

        llama_model = self._models["LLaVA"].model.language_model
        llama_tokenizer = self._models["LLaVA"].processor.tokenizer
        self._models["LLaMA"] = LLaMAModel(model=llama_model,
                                           tokenizer=llama_tokenizer,
                                           device=self.device)


    def _set_seed(self, seed: int) -> None:
        """
        Sets the seed for the random number generators.

        Args:
        seed (int): The seed to use.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


    def to(self, device):
        """
        Moves the model to the specified device.
        
        Args:
        device (torch.device): The device on which the model will run.
        """
        self._models["LLaMA"].to(device)
        self._models["LLaVA"].to(device)
        self._models["PowerPaint"].to(device)
        self.device = device


    def __call__(self,
                image: Image.Image,
                mask: Image.Image,
                current_object: str,
                new_objects_list: Optional[List[str]] = None,
                ddim_steps: int = 50,
                guidance_scale: int = 5,
                seed: int = 1,
                prompt_extension: bool = True, 
                return_prompt: bool = False
                ) -> Tuple[Image.Image, Optional[Tuple[str, str]]] :
                
        """
        Replaces the specified object in the given image with a new one.

        Args:
        image (Image.Image): The input image.
        mask (Image.Image): The mask of the object to replace.
        current_object (str): The name of the object to be replaced.
        new_objects_list (Optional[List[str]]): A list of potential new objects. If None, the method will generate a new object.
        ddim_steps (int): The number of denoising steps. More steps mean a slower but potentially higher quality result.
        guidance_scale (int): The scale for classifier-free guidance. Higher values lead to results that are more closely linked to the text prompt.
        seed (int): Integer value that initializes the random number generator for reproducibility.
        return_prompt (bool): If True, the method also returns the prompt used for generation and the new object.

        Returns:
        Tuple[Image.Image, Optional[Tuple[str, str]]]: The modified image and, optionally, the prompt used for generation and the new object.
        """
        self._set_seed(seed)
        if image.mode != 'RGB': 
            image = image.convert('RGB') 
 
        if mask.mode != 'L': 
            mask = mask.convert('L')

        if prompt_extension:
            image_description = self._models["LLaVA"].generate_image_description(image)
            prompt, new_object = self._models["LLaMA"].generate_prompt(current_object, image_description, new_objects_list)
        else:
            prompt = new_objects_list[0]
            new_object = prompt
        input_image = {'image': image, 'mask': mask}
        
        result = self._models["PowerPaint"](input_image=input_image,
                                            prompt=prompt,
                                            fitting_degree=1.0,
                                            ddim_steps=ddim_steps,
                                            seed=seed,
                                            scale=guidance_scale)
        if return_prompt:
            return result, (prompt, new_object)
        else:
            return result