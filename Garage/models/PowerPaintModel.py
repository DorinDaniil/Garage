import torch
import numpy as np
import os
from transformers import CLIPTextModel
from diffusers import UniPCMultistepScheduler
from .PowerPaint.models import BrushNetModel
from .PowerPaint.models import UNet2DConditionModel
from .PowerPaint.pipelines import StableDiffusionPowerPaintBrushNetPipeline
from .PowerPaint.utils import TokenizerWrapper, add_tokens
from safetensors.torch import load_model
from typing import Dict
from PIL import Image

class PowerPaintModel:
    def __init__(self, 
                 device: str = "cuda", 
                 model_name: str = "ppt-v2-1"):
        """
        Initializes the PowerPaint model.

        Args:
        device (str): Describing the device on which the model will run. Defaults to "cuda".
        checkpoints_path (str): Path to the model checkpoints. Defaults to "ppt-v2-1".
        """
        self.device = torch.device(device)
        script_directory = os.path.dirname(os.path.abspath(__file__))
        self.checkpoints_path = os.path.join(script_directory, "checkpoints", model_name)
        self.weight_dtype = torch.float16
        self.pipe = self._prepare_pipe()


    def _prepare_pipe(self) -> StableDiffusionPowerPaintBrushNetPipeline:
        """
        Prepares the PowerPaint pipeline.

        Returns:
        StableDiffusionPowerPaintBrushNetPipeline: The prepared pipeline.
        """
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            revision=None,
            torch_dtype=self.weight_dtype
        )
        
        text_encoder_brushnet = CLIPTextModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="text_encoder",
            revision=None,
            torch_dtype=self.weight_dtype
        )
        
        brushnet = BrushNetModel.from_unet(unet)
        base_model_path = os.path.join(self.checkpoints_path, "realisticVisionV60B1_v51VAE")
        pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
            base_model_path,
            brushnet=brushnet,
            text_encoder_brushnet=text_encoder_brushnet,
            torch_dtype=self.weight_dtype,
            low_cpu_mem_usage=False,
            safety_checker=None
        )
        
        pipe.unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
            revision=None,
            torch_dtype=self.weight_dtype
        )
        
        pipe.tokenizer = TokenizerWrapper(
            from_pretrained=base_model_path,
            subfolder="tokenizer",
            revision=None,
            torch_type=self.weight_dtype
        )
        
        add_tokens(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder_brushnet,
            placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens=["a", "a", "a"],
            num_vectors_per_token=10
        )
        
        load_model(
            pipe.brushnet,
            os.path.join(self.checkpoints_path, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors")
        )
        
        pipe.text_encoder_brushnet.load_state_dict(
            torch.load(os.path.join(self.checkpoints_path, "PowerPaint_Brushnet/pytorch_model.bin")), strict=False
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)
        return pipe
    

    def to(self, device):
        """
        Moves the model to the specified device.
        
        Args:
        device (torch.device): The device on which the model will run.
        """
        self.pipe.to(device)
        self.device = device


    def __call__(
        self,
        input_image: Dict[str, Image.Image],
        prompt: str,
        fitting_degree: float,
        ddim_steps: int,
        seed: int,
        scale: float) -> Image.Image:
        """
        Makes a prediction using the PowerPaint model.

        Args:
        input_image (Dict[str, Image.Image]): Dictionary with picture and mask for inpainting.
        prompt (str): Prompt for the model.
        fitting_degree (float): Fitting degree for the model.
        ddim_steps (int): The number of denoising steps. More steps mean a slower but potentially higher quality result.
        scale (float): The scale for classifier-free guidance. Higher values lead to results that are more closely linked to the text prompt.

        Returns:
        Image.Image: The predicted image.
        """
        
        size1, size2 = input_image["image"].convert("RGB").size

        if size1 < size2:
            input_image["image"] = input_image["image"].convert("RGB").resize((640, int(size2 / size1 * 640)))
        else:
            input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 640), 640))
            
        promptA = " P_obj"
        promptB = " P_obj"
        negative_promptA = "P_obj"
        negative_promptB = "P_obj"
    
        img = np.array(input_image["image"].convert("RGB"))
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W))
        input_image["image_orig"] = input_image["image"].copy()
        input_image["mask"] = input_image["mask"].resize((H, W))
    
        np_inpimg = np.array(input_image["image"])
        np_inmask = np.array(input_image["mask"]) / 255.0
        np_inpimg = (np_inpimg.transpose(2, 0, 1) * (1 - np_inmask)).transpose(1, 2, 0)
        
        negative_prompt = ("text, bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, "
                    "duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, "
                    "low quality, low res, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality")
        
        result = self.pipe(
            promptA=promptA,
            promptB=promptB,
            promptU=prompt,
            tradoff=fitting_degree,
            tradoff_nag=fitting_degree,
            image=Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB").convert("RGB"),
            mask=input_image["mask"].convert("RGB"),
            num_inference_steps=ddim_steps,
            generator=torch.Generator(self.device).manual_seed(seed),
            brushnet_conditioning_scale=1.0,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            negative_promptU=negative_prompt,
            guidance_scale=scale,
            width=H,
            height=W,
        ).images[0]
        result = result.resize((size1, size2))
        
        return result