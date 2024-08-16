import torch
import transformers
import os
from PIL import Image

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
                 model_name: str = "llava-1.5-7b-hf"):
        """
        Initializes the model.
        
        Args:
        device (str): Describing the device on which the model will run. Defaults to "cuda".
        model_name (str): The name of the model. Defaults to "llava-1.5-7b-hf".
        """
        self.device = torch.device(device)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        checkpoints_directory = os.path.join(script_directory, "checkpoints", model_name)

        self.model = transformers.LlavaForConditionalGeneration.from_pretrained(checkpoints_directory, 
                                                                                torch_dtype=torch.float16
                                                                                ).to(self.device)
        self.processor = transformers.AutoProcessor.from_pretrained(checkpoints_directory)

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
        prompt = "USER: <image>\nI gave you an image. What do you see there? Give me an answer in two or three sequences. ASSISTANT: "
        answer = self._infer(prompt, image)
        answer = answer[answer.rfind('ASSISTANT:') + 10:]
        return answer