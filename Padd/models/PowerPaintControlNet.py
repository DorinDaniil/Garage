import torch
import numpy as np
import os
import random
import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from controlnet_aux import HEDdetector, OpenposeDetector
from .PowerPaint.pipelines import StableDiffusionControlNetInpaintPipeline
from .PowerPaint.pipelines import StableDiffusionInpaintPipeline as Pipeline
from .PowerPaint.utils import TokenizerWrapper, add_tokens
from safetensors.torch import load_model
from typing import Dict, Optional, Tuple
from PIL import Image, ImageFilter

def set_seed(seed: int) -> None:
    """
    Sets the seed for all random number generators.

    Args:
    seed (int): The seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class PowerPaintControlNet:
    def __init__(self, 
                 device: str = "cuda", 
                 model_name: str = "ppt-v1") -> None:
        """
        Initializes the PowerPaint model with ControlNet.

        Args:
        device (str): Describing the device on which the model will run. Defaults to "cuda".
        model_name (str): The name of the model. Defaults to "ppt-v1".
        """
        self.device = device
        script_directory = os.path.dirname(os.path.abspath(__file__))
        self.checkpoints_path = os.path.join(script_directory, "checkpoints", model_name)
        self.weight_dtype = torch.float16
        self.control_pipe = self._prepare_pipe()


    def _prepare_pipe(self) -> StableDiffusionControlNetInpaintPipeline:
        """
        Prepares the PowerPaint pipeline.

        Returns:
        StableDiffusionControlNetInpaintPipeline: The prepared pipeline.
        """
        pipe = Pipeline.from_pretrained("botp/stable-diffusion-v1-5-inpainting", 
                                                torch_dtype=self.weight_dtype,
                                                safety_checker=None,
                                                requires_safety_checker=False     
        )#benjamin-paine/stable-diffusion-v1-5-inpainting
        pipe.tokenizer = TokenizerWrapper(
                from_pretrained="runwayml/stable-diffusion-v1-5",
                subfolder="tokenizer",
                revision=None,
        )

        # add learned task tokens into the tokenizer
        add_tokens(
                tokenizer=pipe.tokenizer,
                text_encoder=pipe.text_encoder,
                placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
                initialize_tokens=["a", "a", "a"],
                num_vectors_per_token=10,
        )

        # loading pre-trained weights
        load_model(pipe.unet, os.path.join(self.checkpoints_path, "unet/unet.safetensors"))
        load_model(pipe.text_encoder, os.path.join(self.checkpoints_path, "text_encoder/text_encoder.safetensors"), strict=False)
        pipe = pipe.to(self.device)

        # initialize controlnet-related models
        self.depth_estimator = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to("cuda")
        self.feature_extractor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        self.hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")

        base_control = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth", torch_dtype=self.weight_dtype
        )
        control_pipe = StableDiffusionControlNetInpaintPipeline(
                pipe.vae,
                pipe.text_encoder,
                pipe.tokenizer,
                pipe.unet,
                base_control,
                pipe.scheduler,
                None,
                None,
                False,
        )
        control_pipe = control_pipe.to(self.device)

        self.current_control = "depth"
        # controlnet_conditioning_scale = 0.8
        return control_pipe
    

    def to(self, device: torch.device) -> None:
        """
        Moves the model to the specified device.
        
        Args:
        device (torch.device): The device on which the model will run.
        """
        self.control_pipe.to(device)
        self.device = device


    def get_depth_map(self, image: Image.Image) -> Image.Image:
        """
        Gets the depth map of the given image.

        Args:
        image (Image.Image): The input image.

        Returns:
        Image.Image: The depth map of the input image.
        """
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad(), torch.autocast(self.device):
            depth_map = self.depth_estimator(image).predicted_depth

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
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image


    def load_controlnet(self, control_type: str) -> None:
        """
        Loads the ControlNet model based on the given control type.

        Args:
        control_type (str): The type of control to load.
        """
        if self.current_control != control_type:
            if control_type == "canny" or control_type is None:
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny", torch_dtype=self.weight_dtype
                )
            elif control_type == "pose":
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-openpose",
                    torch_dtype=self.weight_dtype,
                    local_files_only=self.local_files_only,
                )
            elif control_type == "depth":
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-depth", torch_dtype=self.weight_dtype
                )
            else:
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-hed", torch_dtype=self.weight_dtype
                )
            self.control_pipe = self.control_pipe.to(self.device)
            self.current_control = control_type


    def __call__(
        self,
        input_image: Dict[str, Image.Image],
        control_type: str,
        prompt: str,
        ddim_steps: int,
        scale: float,
        seed: int,
        controlnet_conditioning_scale: float,
        input_control_image: Optional[Image.Image] = None,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Calls the model to generate an image based on the given input.

        Args:
        input_image (Dict[str, Image.Image]): The input image.
        control_type (str): The type of control to use.
        prompt (str): The prompt to use.
        ddim_steps (int): The number of DDIM steps to use.
        scale (float): The scale to use.
        seed (int): The seed to use.
        controlnet_conditioning_scale (float): The ControlNet conditioning scale to use.
        input_control_image (Optional[Image.Image]): The input control image. Defaults to None.

        Returns:
        Tuple[Image.Image, Image.Image]: The generated image and the control image.
        """
        
        promptA = prompt + " P_obj"
        promptB = prompt + " P_obj"
        negative_promptA = "worst quality, low quality, normal quality, bad quality, blurry, P_obj"
        negative_promptB = "worst quality, low quality, normal quality, bad quality, blurry, P_obj"

        size1, size2 = input_image["image"].convert("RGB").size

        if size1 < size2:
            input_image["image"] = input_image["image"].convert("RGB").resize((640, int(size2 / size1 * 640)))
        else:
            input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 640), 640))
        img = np.array(input_image["image"].convert("RGB"))
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W))
        input_image["mask"] = input_image["mask"].resize((H, W))

        if control_type != self.current_control:
            self.load_controlnet(control_type)
        if input_control_image is None:
            controlnet_image = input_image["image"]
        else:
            controlnet_image = input_control_image
        if control_type == "canny":
            controlnet_image = controlnet_image.resize((H, W))
            controlnet_image = np.array(controlnet_image)
            controlnet_image = cv2.Canny(controlnet_image, 100, 200)
            controlnet_image = controlnet_image[:, :, None]
            controlnet_image = np.concatenate([controlnet_image, controlnet_image, controlnet_image], axis=2)
            controlnet_image = Image.fromarray(controlnet_image)
        elif control_type == "pose":
            controlnet_image = self.openpose(controlnet_image)
        elif control_type == "depth":
            controlnet_image = controlnet_image.resize((H, W))
            if input_control_image is None:
                controlnet_image = self.get_depth_map(controlnet_image)
        else:
            controlnet_image = self.hed(controlnet_image)

        controlnet_image = controlnet_image.resize((H, W))
        set_seed(seed)
        result = self.control_pipe(
            promptA=promptB,
            promptB=promptA,
            tradoff=1.0,
            tradoff_nag=1.0,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            image=input_image["image"].convert("RGB"),
            mask=input_image["mask"].convert("RGB"),
            control_image=controlnet_image,
            width=H,
            height=W,
            guidance_scale=scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=ddim_steps,
        ).images[0]

        m_img = input_image["mask"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=4))
        m_img = np.asarray(m_img) / 255.0
        img_np = np.asarray(input_image["image"].convert("RGB")) / 255.0
        ours_np = np.asarray(result) / 255.0
        ours_np = ours_np * m_img + (1 - m_img) * img_np
        result_paste = Image.fromarray(np.uint8(ours_np * 255))

        result_paste = result_paste.resize((size1, size2))
        controlnet_image = controlnet_image.resize((size1, size2))
        return result_paste, controlnet_image