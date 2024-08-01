import cv2
import numpy as np
import torch
import time
from PIL import Image
import gradio as gr

from GenerativeAugmentations.models.GroundedSegmentAnything.segment_anything.segment_anything import SamPredictor, sam_model_registry
from GenerativeAugmentations.models.GroundedSegmentAnything.GroundingDINO.groundingdino.util.inference import Model
from GenerativeAugmentations import Augmenter

MODEL_DICT = dict(
    vit_h="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",  # yapf: disable  # noqa
    vit_l="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",  # yapf: disable  # noqa
    vit_b="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",  # yapf: disable  # noqa
)

GROUNDING_DINO_CONFIG_PATH = "GenerativeAugmentations/models/GroundedSegmentAnything/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "GenerativeAugmentations/models/GroundedSegmentAnything/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = "GenerativeAugmentations/models/GroundedSegmentAnything/sam_vit_h_4b8939.pth"
SAM_ENCODER_VERSION = "vit_h"

class GradioWindow():
    def __init__(self) -> None:
        self.saved_points = []
        self.saved_labels = []

        self.GROUNDING_DINO_CONFIG_PATH = GROUNDING_DINO_CONFIG_PATH
        self.GROUNDING_DINO_CHECKPOINT_PATH = GROUNDING_DINO_CHECKPOINT_PATH
        self.model_type = SAM_ENCODER_VERSION
        self.SAM_CHECKPOINT_PATH = SAM_CHECKPOINT_PATH

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #self.augmenter = None
        self.augmenter = Augmenter(device=self.device)
        self.predictor = self.setup_model()
        self.main()

    def main(self):
        with gr.Blocks() as self.demo:
            with gr.Row():
                input_img = gr.Image(type="pil", label="Input image", interactive=True)
                selected_mask = gr.Image(type="pil", label="Selected Mak")
                segmented_img = gr.Image(type="pil", label="Selected Segment")

            with gr.Row():
                with gr.Group():
                    self.current_object = gr.Textbox(label="Current object", value="The running dog")
                    with gr.Accordion("Advanced options", open=False):
                        box_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Box threshold")
                        text_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Text threshold")

                    segment_object = gr.Button("Segment object")
                
                with gr.Column(): 
                    gr.Examples(
                        label="Images Examples",
                        examples=[
                        ["examples/dog.jpg"],
                        ["examples/bread.png"],
                        ["examples/room.jpg"],
                        ["examples/spoon.png"],
                        ["examples/image.jpg"], 
                        ], 
                        inputs=[input_img],
                        examples_per_page=5      
                    )
                    gr.Examples(
                        label="Mask Examples",
                        examples=[
                        ["examples/dog_mask.jpg"],
                        ["examples/bread_mask.jpg"],
                        ["examples/room_mask.jpg"],
                        ["examples/spoon_mask.jpg"],
                        ["examples/image_mask.jpg"], 
                        ], 
                        inputs=[selected_mask, input_img],    
                        outputs=[segmented_img],
                        fn=self.show_mask,
                        run_on_click=True
                    )

            with gr.Row():
                with gr.Column(): 
                    with gr.Group():
                        self.target_object = gr.Textbox(label="Target object", value="dog")

                        with gr.Accordion("Generation options", open=False):
                            self.iter_number = gr.Number(value=50, label="Steps")
                            self.guidance_scale = gr.Number(value=5, label="Guidance Scale")
                            self.seed = gr.Number(value=1, label="Seed")
                            self.return_prompt = gr.Checkbox(value=True, label="Show generated prompt")

                        enter_prompt = gr.Button("Augment Image")

                with gr.Column():
                    augmented_img = gr.Image(type="pil", label="Augmented Image")
                    generated_prompt = gr.Markdown(
                            f"<div class=\"message\" style=\"text-align: center; \
                                font-size: 18px;\"></div>", 
                            visible=True)

            # Connect the UI and logic
            segment_object.click(
                self.detect,
                inputs=[input_img, self.current_object, box_threshold, text_threshold],
                outputs=[segmented_img, selected_mask]
            )

            enter_prompt.click(
                self.augment_image,
                inputs=[input_img, self.current_object, self.target_object, 
                        self.iter_number, self.guidance_scale, self.seed, self.return_prompt],
                outputs=[augmented_img, generated_prompt],
            )

    def setup_model(self) -> SamPredictor:
        self.sam = sam_model_registry[self.model_type](checkpoint=self.SAM_CHECKPOINT_PATH)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)

        self.grounding_dino_model = Model(
            model_config_path=self.GROUNDING_DINO_CONFIG_PATH, 
            model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT_PATH, 
            device=self.device
            )

    def detect(self, image: Image, prompt: str, box_threshold: float, text_threshold: float):
        detections = self.grounding_dino_model.predict_with_classes(
            image=cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB),
            classes=[prompt],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        detections.mask = self.segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        mask = Image.fromarray(detections.mask[0])
        res = self.show_mask(mask, image)
        return res, mask
    
    def show_mask(self, mask: Image, image: Image, 
                  random_color: bool = False) -> np.ndarray:
        """Visualize a mask on top of an image.
        Args:
            mask (Image): A 2D array of shape (H, W, 3).
            image (Image): A 3D array of shape (H, W, 3).
            random_color (bool): Whether to use a random color for the mask.
        Returns:
            np.ndarray: A 3D array of shape (H, W, 3) with the mask
            visualized on top of the image.
        """
        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255])
        
        mask, image = np.array(mask.convert('L')), np.array(image)
        mask = np.where(mask > 200, 1, 0).astype(np.uint8)

        target_size = (image.shape[1], image.shape[0])  # width, height
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255
        image = cv2.addWeighted(image, 0.7, mask_image.astype("uint8"), 0.3, 0)
        return image

    def show_points(self, coords: np.ndarray, 
                    labels: np.ndarray, image: Image) -> np.ndarray:
        """Visualize points on top of an image.
        Args:
            coords (np.ndarray): A 2D array of shape (N, 2).
            labels (np.ndarray): A 1D array of shape (N,).
            image (np.ndarray): A 3D array of shape (H, W, 3).
        Returns:
            np.ndarray: A 3D array of shape (H, W, 3) with the points
            visualized on top of the image.
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        image = np.array(image)
        for p in pos_points:
            image = cv2.circle(
                image, p.astype(int), radius=5, color=(0, 255, 0), thickness=-1
            )
        for p in neg_points:
            image = cv2.circle(
                image, p.astype(int), radius=5, color=(255, 0, 0), thickness=-1
            )
        return image
    
    def segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def augment_image(self, image: np.array, 
                      current_object: str, new_objects_list: list,
                      ddim_steps: int, guidance_scale: int, seed: int, return_prompt: str) -> tuple:
        
        self.masks = self.masks.astype(np.uint8) * 255
        self.masks = np.squeeze(self.masks)
        self.masks = Image.fromarray(self.masks, mode='L')

        result, (prompt, new_object) = self.augmenter(
        image=image,
        mask=self.masks,
        current_object=current_object,
        new_objects_list=new_objects_list,
        ddim_steps=ddim_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        return_prompt=True
        )
        return result
    
if __name__ == "__main__":
    window = GradioWindow()
    window.demo.launch(share=False)
    window.demo.close()