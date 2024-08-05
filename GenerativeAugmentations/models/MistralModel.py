import torch
import transformers
from typing import Optional, List, Tuple

class MistralModel:
    """
    A class implementing the Mistral model for generating responses based on text input.

    Attributes:
    device (torch.device): The device on which the model will run.
    model (transformers.AutoModelForCausalLM): The Mistral model for causal language modeling.
    tokenizer (transformers.AutoTokenizer): The tokenizer for handling input data.
    generation_params (dict): The parameters for generating responses.
    """

    def __init__(self, 
                 device: str = "cuda", 
                 model_name: str = "Intel/neural-chat-7b-v3-1"):
        """
        Initializes the model.

        Args:
        device (str): Describing the device on which the model will run. Defaults to "cuda".
        model_name (str): The name of the model. Defaults to "Intel/neural-chat-7b-v3-1".
        """
        self.device = torch.device(device)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.model = self.model.to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
        self.generation_params = {
            "do_sample": True,
            "temperature": 1,
            "top_p": 0.90,
            "top_k": 40,
            "max_new_tokens": 77,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    def _infer(self, 
               prompt: str) -> str:
        """
        Generates a response based on the given prompt.

        Args:
        prompt (str): The text input.

        Returns:
        str: The generated response.
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(self.device)

        outputs = self.model.generate(inputs, num_return_sequences=1, **self.generation_params)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

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
                    " You should select and return only the name of new object from the provided list,"
                    " which fits into the picture to replace the existing one. ASSISTANT: ")

        new_object = self._infer(prompt_1 if new_objects_list is None else prompt_2)
        new_object = new_object[new_object.rfind('ASSISTANT:') + 10:]
        return ' '.join(new_object.split()).lower()

    def generate_prompt(self, 
                        current_object: str, 
                        image_description: str, 
                        new_objects_list: Optional[List[str]] = None,
                        return_new_object: bool = False) -> Tuple[str, str]:
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
                f" Focus solely on the realistic description of the {new_object}, ignoring any external elements or surroundings."
                " For example, if the object is an animal, the description should include the animal's color, size, breed, pose, view direction etc."
                " If the object is a vehicle, the description should include vehicle's brand or model, color, size, type, etc."
                " If the object is a person, the description should include person's age, gender, height, weight, hair color, eye color, clothing, pose, etc."
                f" Do not add anything extra to the visual description that is not directly related to {new_object}. ASSISTANT: ")

        output_prompt = self._infer(prompt)
        output_prompt = output_prompt[output_prompt.rfind('ASSISTANT:') + 10:]
        new_object = new_object.replace(' ', '')
        return ' '.join(output_prompt.split()), new_object