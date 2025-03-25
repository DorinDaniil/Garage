import torch
import transformers
import os
from PIL import Image

from transformers import LlavaForConditionalGeneration, AutoProcessor

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
                 model_name: str = "llava-llama-3-8b"):
        """
        Initializes the model.
        
        Args:
        device (str): Describing the device on which the model will run. Defaults to "cuda".
        model_name (str): The name of the model. Defaults to "llava-llama-3-8b".
        """
        self.device = torch.device(device)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        checkpoints_directory = os.path.join(script_directory, "checkpoints", model_name)

        self.model = LlavaForConditionalGeneration.from_pretrained(checkpoints_directory, 
                                                        torch_dtype=torch.float16,
                                                        low_cpu_mem_usage=True
                                                        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(checkpoints_directory)

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
        answer = self.processor.decode(generate_ids[0][2:], skip_special_tokens=True)
        return answer


    def to(self, device):
        """
        Moves the model to the specified device.
        
        Args:
        device (torch.device): The device on which the model will run.
        """
        self.device = device
        self.model.to(device)


    def generate_image_description(self, 
                              image: Image.Image) -> str:
        """
        Generates a description of the given image.
        
        Args:
        image (Image.Image): The image to describe.
        
        Returns:
        str: The generated description.
        """
        prompt = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat are these?<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n")

        answer = self._infer(prompt, image)
        answer = answer[answer.rfind('assistant\n\n') + 11:]
        return answer