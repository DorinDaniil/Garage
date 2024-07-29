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

        self.augmenter = None
        # self.augmenter = Augmenter(device=self.device)
        self.setup_model()
        self.main()

    def main(self):
        with gr.Blocks() as self.demo:
            with gr.Row():
                with gr.Group():
                    input_img = gr.Image(label="Input image", interactive=True)
                    self.current_object = gr.Textbox(label="Current object", value="The running dog")
                    with gr.Accordion("Advanced options", open=False):
                        box_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Box threshold")
                        text_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Text threshold")

                    segment_object = gr.Button("Segment object")

                with gr.Group():
                    segmented_img = gr.Image(label="Selected Segment")
                    loading_status = gr.Markdown(
                        "<div class=\"message\" style=\"text-align: center; \
                            font-size: 24px;\">Please load image</div>", 
                        visible=True)

            with gr.Row():
                with gr.Column(): 
                    self.target_object = gr.Textbox(label="Target object", value="dog")

                    self.iter_number = gr.Number(value=50, label="Steps")
                    self.guidance_scale = gr.Number(value=5, label="Guidance Scale")
                    self.seed = gr.Number(value=1, label="Seed")

                    enter_prompt = gr.Button("Augment Image")
                    
                    reset = gr.Button("Reset Points")

                with gr.Column():
                    augmented_img = gr.Image(label="Augmented Image")

            # Connect the UI and logic
            segment_object.click(
                self.detect,
                inputs=[input_img, self.current_object, box_threshold, text_threshold],
                outputs=[segmented_img]
            )

            enter_prompt.click(
                self.augment_image,
                inputs=[input_img, self.current_object, self.target_object, 
                        self.iter_number, self.guidance_scale, self.seed],
                outputs=[augmented_img],
            )

            reset.click(self.reset_points)

    def setup_model(self) -> SamPredictor:
        self.sam = sam_model_registry[self.model_type](checkpoint=self.SAM_CHECKPOINT_PATH)
        # self.sam.load_state_dict(torch.utils.model_zoo.load_url(MODEL_DICT[self.model_type]))
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)

        self.grounding_dino_model = Model(
            model_config_path=self.GROUNDING_DINO_CONFIG_PATH, 
            model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT_PATH, 
            device=self.device
            )

    def set_image(self, img) -> None:
        """Set the image for the predictor."""
        self.predictor.set_image(img)
        print("Image loaded!")
        return "<div class=\"message\" style=\"text-align: center; font-size: 24px;\">Image Loaded!</div>"

    def detect(self, image: Image, prompt: str, box_threshold: float, text_threshold: float):
        detections = self.grounding_dino_model.predict_with_classes(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            classes=[prompt],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        detections.mask = self.segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        mask = detections.mask[0]
        res = self.show_mask(mask, image)
        return res
    
    def show_mask(self, mask: np.ndarray, image: np.ndarray, 
                  random_color: bool = False) -> np.ndarray:
        """Visualize a mask on top of an image.
        Args:
            mask (np.ndarray): A 2D array of shape (H, W).
            image (np.ndarray): A 3D array of shape (H, W, 3).
            random_color (bool): Whether to use a random color for the mask.
        Returns:
            np.ndarray: A 3D array of shape (H, W, 3) with the mask
            visualized on top of the image.
        """
        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255

        image = cv2.addWeighted(image, 0.7, mask_image.astype("uint8"), 0.3, 0)
        return image

    def show_points(self, coords: np.ndarray, 
                    labels: np.ndarray, image: np.ndarray) -> np.ndarray:
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

    def sleep(self, input_img):
        original_img = input_img["background"]
        mask = input_img["layers"][0]
        mask = np.array(Image.fromarray(np.uint8(mask)).convert("L"))
        masks = np.where(mask != 0, 255, 0)
        return [original_img, masks, input_img["composite"]]

    def reset_points(self) -> None:
        """Reset the points."""
        self.saved_points = []
        self.saved_labels = []

    def augment_image(self, image: np.array, 
                      current_object: str, new_objects_list: list,
                      ddim_steps: int, guidance_scale: int, seed: int) -> tuple:
        
        print("SEGMENTATION MASK: ", self.masks.shape, type(self.masks), np.unique(self.masks))
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