import torch
import transformers
import os
from tqdm import tqdm
from requests import get
from urllib.parse import urlencode
from math import ceil
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple

from MistralModel import MistralModel
from PowerPaintPipeline import StableDiffusionInpaintPipeline
from PowerPaintPipeline import TokenizerWrapper, add_tokens
from safetensors.torch import load_model
    
class augmenter():
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.mistral = MistralModel()
        self.llava = transformers.LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf").to(self.device)
        self.llava_processor = transformers.AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        weight_dtype = torch.float16
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", 
                                                                    torch_dtype=weight_dtype,
                                                                    safety_checker=None,
                                                                    requires_safety_checker=False)
        self.pipe.tokenizer = TokenizerWrapper(
            from_pretrained="runwayml/stable-diffusion-v1-5", subfolder="tokenizer", revision=None
        )

        add_tokens(
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder,
            placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
        )
        
        self._download_unet_model_weights()
        load_model(self.pipe.unet, "./unet/unet.safetensors")
        load_model(self.pipe.text_encoder, "./unet/text_encoder.safetensors", strict=False)
        self.pipe = self.pipe.to(self.device)

    def _download_unet_model_weights(self,
                                model_path='unet',
                                public_key_unet='https://disk.yandex.ru/d/_z_j4XbMBpo9iA',
                                public_key_text_encoder='https://disk.yandex.ru/d/S9yNXtO-sxu_DQ'):

        if not os.path.exists(model_path):
            os.makedirs(model_path)
    
        unet_weights_path = os.path.join(model_path, 'unet.safetensors')
        text_encoder_weights_path = os.path.join(model_path, 'text_encoder.safetensors')
    
        if not (os.path.exists(unet_weights_path) and os.path.exists(text_encoder_weights_path)):
            base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    
            final_url_unet = base_url + urlencode(dict(public_key=public_key_unet))
            response_unet = get(final_url_unet, stream=True)
            download_url_unet = response_unet.json()['href']
    
            final_url_text_encoder = base_url + urlencode(dict(public_key=public_key_text_encoder))
            response_text_encoder = get(final_url_text_encoder, stream=True)
            download_url_text_encoder = response_text_encoder.json()['href']
    
            with get(download_url_unet, stream=True) as download_response_unet:
                with open(unet_weights_path, 'wb') as unet_file:
                    for chunk in tqdm(iterable=download_response_unet.iter_content(1024),
                                       desc="Downloading UNET weights",
                                       unit="KB",
                                       unit_scale=True):
                        unet_file.write(chunk)
    
            with get(download_url_text_encoder, stream=True) as download_response_text_encoder:
                with open(text_encoder_weights_path, 'wb') as text_encoder_file:
                    for chunk in tqdm(iterable=download_response_text_encoder.iter_content(1024),
                                       desc="Downloading Text Encoder weights",
                                       unit="KB",
                                       unit_scale=True):
                        text_encoder_file.write(chunk)
    
    def _resize_image_div(self, pil_image: Image.Image, size: int = 8) -> Image.Image:
        # Get original image dimensions
        height = pil_image.size[0]
        width = pil_image.size[1]
        # Calculate nearest multiples of 8 for each dimension
        new_height = int((ceil(height / size)) * size)
        new_width = int((ceil(width / size)) * size)
        return pil_image.resize((new_height, new_width))

    def _get_output_mistral(self, prompt: str) -> str:
        answer = self.mistral.infer_prompt(prompt)
        answer = answer[answer.rfind('ASSISTANT:') + 10:]
        return ' '.join(answer.split())

    def _get_output_llava(self, prompt: str, image: Image.Image) -> str:
        inputs = self.llava_processor(text=prompt, images=image, return_tensors="pt")
        for elem in inputs:
            inputs[elem] = inputs[elem].to(self.device)

        generate_ids = self.llava.generate(**inputs, max_new_tokens=50)
        answer = self.llava_processor.batch_decode(generate_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)[0]
        answer = answer[answer.rfind('ASSISTANT:') + 10:]
        return answer

    def _generate_prompt_llava_mistral(self,
                                        pil_image: Image.Image,
                                        current_object: str,
                                        new_objects_list: Optional[List[str]] = None) -> Tuple[str, str]:
        PROMPT0 = "USER: <image>\nI gave you an image. What do you see there? Give me an answer in two or three sequences. ASSISTANT: "
        image_description = self._get_output_llava(PROMPT0, pil_image)

        PROMPT1v1 = "USER: Imagine you are a object replacer. Your task is generating a replacement object instead of the existing object on the "\
                "scene. It's important that the new object is not the same as the existing one. I will give you a description of the scene and "\
                "the existing object. You must give me an object which could be "\
                f"depicted instead of existing object. So, image description: {image_description}, existing object: {current_object}. You should return "\
                "only a name of new object and nothing else. ASSISTANT: a"

        PROMPT1v2 = "USER: Imagine you are a object replacer. Your task is generating a replacement object instead of the existing object on the "\
                "scene. It's important that the new object is not the same as the existing one. "\
                "I will give you a description of the scene, existing object and a list of potential new objects. "\
                "You must give me an object from the list of potential new objects which could be depicted instead of existing object."\
                f"So, image description: {image_description}, existing "\
                f"object: {current_object}, a list of potential new objects: {new_objects_list}. "\
                "You should select and return only the name of new object from the provided list, "\
                "which is different from the existing one. ASSISTANT: "

        new_object = self._get_output_mistral(PROMPT1v1 if new_objects_list is None else PROMPT1v2).lower()
      
        if new_object.startswith('a '):
            new_object_return = new_object[2:].replace(' ', '')
        else:
            new_object_return = new_object.replace(' ', '')

        PROMPT2 = f"USER: Imagine that you want to describe the {new_object}'s appearance to an artist in one sentence, under 15 words. "\
                f"Mention {new_object} in the description for clarity. "\
                f"Focus solely on the realistic description of the {new_object}, ignoring any external elements or surroundings. "\
                "For example, if the object is an animal, the description should include the animal's color, size, breed, pose, view direction etc. "\
                "If the object is a vehicle, the description should include vehicle's brand or model, color, size, type, etc. "\
                "If the object is a person, the description should include person's age, gender, height, weight, hair color, "\
                f"eye color, clothing, pose, etc. "\
                f"Do not add anything extra to the visual description that is not directly related to {new_object}. "\
                "ASSISTANT: "
        
        prompt = self._get_output_mistral(PROMPT2)
        return new_object_return, prompt

    def __call__(self,
                pil_image: Image.Image,
                pil_mask: Image.Image,
                current_object: str,
                new_objects_list: Optional[List[str]] = None,
                num_inference_steps: int = 50,
                guidance_scale: int = 7,
                strength: float = 1.0,
                return_prompt: bool = False) -> Tuple[Image.Image, Optional[Tuple[str, str]]]:
        """
        This method performs the main task of the class: it replaces the specified object in the given image with a new one.

        It first resizes the input image and mask to the nearest multiple of 8 for compatibility with the model's requirements.
        Then, it generates a new object and a prompt using the generate_prompt_llava_mistral method.
        
        After that, it prepares the prompts and negative prompts for the Stable Diffusion Inpainting Pipeline.
        The pipeline then generates a new image where the specified object is replaced with the new one.

        The method returns the modified image and, optionally, the new object and the prompt used for generation.

        Parameters:
            pil_image (Image.Image): The input image.
            pil_mask (Image.Image): The mask of object to replace.
            current_object (str): The name of the object to be replaced.
            new_objects_list (Optional[List[str]]): A list of potential new objects. If None, the method will generate a new object.
            num_inference_steps (int): The number of denoising steps. More steps mean a slower but potentially higher quality result.
            guidance_scale (int): The scale for classifier-free guidance. Higher values lead to results that are more closely linked to the text prompt.
            strength (float, defaults to 1.0):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            return_prompt (bool): If True, the method also returns the new object and the prompt used for generation.
        """
        image_resized = self._resize_image_div(pil_image)
        mask_resized = self._resize_image_div(pil_mask)
        new_object, prompt = self._generate_prompt_llava_mistral(image_resized, current_object, new_objects_list)
        
        negative_prompt = "text, (((bad anatomy, bad proportions))), blurry, cropped, (((deformed))), disfigured, "\
                    "duplicate, error, (((extra limbs))), gross proportions, jpeg artifacts, long neck, "\
                    "low quality, (((low res))), malformed, morbid, mutated, mutilated, out of frame, ugly"

        promptA = prompt #+ " P_obj"
        promptB = prompt #+ " P_obj"
        negative_promptA = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry, P_obj"
        negative_promptB = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry, P_obj"

        results = self.pipe(
            promptA=promptA,
            promptB=promptB,
            image=image_resized,
            mask_image=mask_resized,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            width=image_resized.size[0],
            height=image_resized.size[1],
            strength=1.0,
            num_images_per_prompt=1)

        modified_image = results.images[0].resize((pil_image.size))
        if return_prompt:
            return modified_image, new_object, prompt
        else:
            return modified_image



        