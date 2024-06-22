import os
import torch
import transformers
from math import ceil
from PIL import Image, ImageDraw
from typing import List, Optional, Tuple
from PowerPaintPipeline import StableDiffusionInpaintPipeline as Pipeline
from PowerPaintPipeline import TokenizerWrapper, add_tokens
from safetensors.torch import load_model


def PILmask_from_bboxcords(cords: Tuple[int, int, int, int], img_width: int, img_height: int) -> Image.Image:
        xmin, ymin, xmax, ymax = cords
        # Create black PIL mask
        pil_mask = Image.new('L', (img_width, img_height))
        # Draw white bounding box on the image
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([xmin, ymin, xmax, ymax], fill=255)
        return pil_mask


class MistralModel:
    def __init__(
        self,
        device_str='cuda',
        model_name='Intel/neural-chat-7b-v3-1',
        ):
        
        self.device = torch.device(device_str)
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.generation_params = {
            "do_sample": True,
            "temperature": 1,
            "top_p": 0.90,
            "top_k": 40,
            "max_new_tokens": 256,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
    def infer_prompt(
        self,
        prompt
        ):
        # Tokenize and encode the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(self.device)
        
        # Generate a response
        outputs = self.model.generate(inputs, num_return_sequences=1, **self.generation_params)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        return response
        
    def expand_prompt(
        self,
        prompt,
        max_length=4096
        ):
        system_input = "You are a prompt engineer. Your mission is to expand prompts written by user. You should provide the best prompt "\
                    "for text to image generation in English."
        prompt = f"### System:\n{system_input}\n### User:\n{prompt}\n### Assistant:\n"
        
        response = self.infer_prompt(prompt)
        result = response.split("### Assistant:\n")[-1]
        if len(result) > max_length:
            result = result[:4096]
        return result


class augmenter():
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.mistral = MistralModel()
        self.llava = transformers.LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf").to(self.device)
        self.llava_processor = transformers.AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        weight_dtype = torch.float16
        self.pipe = Pipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=weight_dtype)
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

        load_model(self.pipe.unet, "./unet/unet.safetensors")
        load_model(self.pipe.text_encoder, "./unet/text_encoder.safetensors", strict=False)
        self.pipe = self.pipe.to(self.device)

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

    def generate_prompt_llava_mistral(self,
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
                f"So, image description: {image_description}, existing "\
                f"object: {current_object}, a list of potential objects.: {new_objects_list}. You should select "\
                "and return only the name of the new object from the provided list, which is different from the existing object. ASSISTANT: "

        new_object = self._get_output_mistral(PROMPT1v1 if new_objects_list is None else PROMPT1v2).lower()
      
        if new_object.startswith('a '):
            new_object_return = new_object[2:].replace(' ', '')
        else:
            new_object_return = new_object.replace(' ', '')

        PROMPT2 = "USER: Imagine you are describing the visual appearance of an object using only adjectives. "\
                "Your task is to provide a brief and detailed visual description "\
                "using only adjectives that convey the object's appearance. Do not include any extra words or synonyms for the object's name. "\
                "Then, at the end of the description, include the name of the object. "\
                f"So, оbject: {new_object}. ASSISTANT: a"

        prompt = self._get_output_mistral(PROMPT2)
        return new_object_return, prompt

    def augment(self,
                pil_image: Image.Image,
                pil_mask: Image.Image,
                current_object: str,
                new_objects_list: Optional[List[str]] = None,
                num_inference_steps: int = 200,
                guidance_scale: int = 6,
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
            return_prompt (bool): If True, the method also returns the new object and the prompt used for generation.
        """
        image_resized = self._resize_image_div(pil_image)
        mask_resized = self._resize_image_div(pil_mask)
        new_object, prompt = self.generate_prompt_llava_mistral(image_resized, current_object, new_objects_list)

        negative_prompt = ''
        promptA = prompt + " P_ctxt"
        promptB = prompt + " P_ctxt"

        negative_promptA = negative_prompt + " P_obj"
        negative_promptB = negative_prompt + " P_obj"

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
            height=image_resized.size[1])

        modified_image = results.images[0].resize((pil_image.size))
        if return_prompt:
            return modified_image, new_object, prompt
        else:
            return modified_image



        