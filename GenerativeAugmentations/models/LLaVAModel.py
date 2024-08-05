import torch
import transformers
from PIL import Image
from torch import Tensor
from typing import Dict

class LLaVAModel:
    """
    A class implementing the LLaVA model for generating responses based on text input and an image.
    
    Attributes:
    device (torch.device): The device on which the model will run.
    model (transformers.LlavaForConditionalGeneration): The LLaVA model for conditional text generation.
    processor (transformers.AutoProcessor): The processor for handling input data.
    """

    def __init__(self, 
                 device: str = "cuda", 
                 model_name: str = "llava-hf/llava-1.5-7b-hf"):
        """
        Initializes the model.
        
        Args:
        device (str): Describing the device on which the model will run. Defaults to "cuda".
        model_name (str): The name of the model. Defaults to "llava-hf/llava-1.5-7b-hf".
        """
        self.device = torch.device(device)
        self.model = transformers.LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(self.device)
        self.processor = transformers.AutoProcessor.from_pretrained(model_name, torch_dtype=torch.float16)

    def _infer(self, 
               prompt: str, 
               image: Image.Image) -> str:
        """
        Generates a response based on text input and an image.
        
        Args:
        prompt (str): The text input.
        image (Image.Image): The image.
        
        Returns:
        str: The generated response.
        """
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        for elem in inputs:
            inputs[elem] = inputs[elem].to(self.device)

        generate_ids = self.model.generate(**inputs, max_new_tokens=50)
        answer = self.processor.batch_decode(generate_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)[0]
        return answer

    def generate_image_description(self, 
                              image: Image.Image) -> str:
        """
        Generates a description of the given image.
        
        Args:
        image (Image.Image): The image to describe.
        
        Returns:
        str: The generated description.
        """
        prompt = "USER: <image>\nI gave you an image. What do you see there? Give me an answer in two or three sequences. ASSISTANT: "
        answer = self._infer(prompt, image)
        answer = answer[answer.rfind('ASSISTANT:') + 10:]
        return answer