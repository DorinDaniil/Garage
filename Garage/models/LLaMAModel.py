import torch
import transformers
from typing import Optional, List, Tuple

class LLaMAModel:
    """
    A class implementing the LLaMA model for generating responses based on text input.

    Attributes:
    device (torch.device): The device on which the model will run.
    model (transformers.pipeline): The LLaMA model for causal language modeling.
    generation_params (dict): The parameters for generating responses.
    """

    def __init__(self, 
                 model=None,
                 tokenizer=None,
                 device: str = "cuda"):
        """
        Initializes the model.
        """
        self.device = torch.device(device)

        self.model = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                model_kwargs={"torch_dtype": torch.float16},
                device=self.device,
                pad_token_id=128009
        )
        
        terminators = [
                self.model.tokenizer.eos_token_id,
                self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        self.generation_params = {
                "max_new_tokens": 50,
                "eos_token_id": terminators,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.9
        }


    def to(self, device):
        """
        Moves the model to the specified device.
        
        Args:
        device (torch.device): The device on which the model will run.
        """
        self.model.model.to(device)
        self.device = device


    def _infer(self, 
               prompt: str) -> str:
        """
        Generates a response based on the given prompt.

        Args:
        prompt (str): The text input.

        Returns:
        str: The generated response.
        """
        outputs = self.model(prompt, **self.generation_params)
        response = outputs[0]["generated_text"]
        assistant_response = response.replace(prompt, '', 1).strip().split('\n')[0]

        return assistant_response


    def _generate_new_object(self, 
                             current_object: str, 
                             image_description: str, 
                             new_objects_list: Optional[List[str]] = None) -> str:
        """
        Generates a new object to replace the existing object in the image description.

        Args:
        current_object (str): The existing object.
        image_description (str): The image description.
        new_objects_list (Optional[List[str]], optional): A list of potential new objects. Defaults to None.

        Returns:
        str: The new object.
        """
        prompt_1 = ("USER: Imagine you are a object replacer. Your task is generating a replacement object instead of the existing object on the scene."
                " It's important that the new object is not the same as the existing one. I will give you a description of the scene and the existing object."
                " You must give me an object which could be depicted instead of existing object."
                f" So, image description: {image_description}, existing object: {current_object}."
                " You should return only a name of new object and nothing else. ASSISTANT: a")

        prompt_2 = ("USER: Imagine you are a object replacer. Your task is generating a replacement object instead of the existing object on the scene."
                    " It's important that the new object is not the same as the existing one. "
                    " I will give you a description of the scene, existing object and a list of potential new objects."
                    " You must give me an object from the list of potential new objects which could be depicted instead of existing object."
                    " The new object should fit well into the picture in place of the existing object."
                    " The new object should be approximately the same size as the existing object."
                    " If no object from the list fits into the picture, return the existing object."
                    " The image should remain believable after replacement."
                    f" So, image description: {image_description}, existing object: {current_object}, a list of potential new objects: {new_objects_list}."
                    " You should select and return only the name of new object without quotes from the provided list,"
                    " which fits into the picture to replace the existing one. Return only name of new object and nothing else. ASSISTANT: a")

        new_object = self._infer(prompt_1 if new_objects_list is None else prompt_2)

        return ' '.join(new_object.split()).lower()


    def generate_prompt(self, 
                        current_object: str, 
                        image_description: str, 
                        new_objects_list: Optional[List[str]] = None) -> Tuple[str, str]:
        """
        Generates a prompt with a description of the new object.

        Args:
        current_object (str): The existing object.
        image_description (str): The image description.
        new_objects_list (Optional[List[str]], optional): A list of potential new objects. Defaults to None.
        
        Returns:
        Tuple[str, str]: Prompt and new object.
        """
        
        new_object = self._generate_new_object(current_object, image_description, new_objects_list)
        new_object = new_object[2:] if new_object.startswith('a ') else new_object

        prompt = (f"USER: Imagine that you want to describe the {new_object}'s appearance to an artist in one sentence, under 15 words."
                f" Mention {new_object} in the description for clarity."
                " The description should be concise, using no more than 15 words."
                f" Focus solely on the realistic description of the {new_object}, ignoring any external elements or surroundings."
                " For example, if the object is an animal, the description should include the animal's color, size, breed, pose, view direction etc."
                " If the object is a vehicle, the description should include vehicle's brand or model, color, size, type, etc."
                " If the object is a person, the description should include person's age, gender, height, weight, hair color, eye color, clothing, pose, etc."
                f" Do not add anything extra to the visual description that is not directly related to {new_object}. ASSISTANT: ")

        output_prompt = self._infer(prompt)
        return ' '.join(output_prompt.split()), new_object