import cv2
import numpy as np
import torch
import time
from PIL import Image, ImageDraw
import gradio as gr
import matplotlib.pyplot as plt

from Garage.models.GroundedSegmentAnything.segment_anything.segment_anything import SamPredictor, sam_model_registry
from Garage.models.GroundedSegmentAnything.GroundingDINO.groundingdino.util.inference import Model
from Garage import Augmenter


MODEL_DICT = dict(
    vit_h="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",  # yapf: disable  # noqa
    vit_l="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",  # yapf: disable  # noqa
    vit_b="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",  # yapf: disable  # noqa
)

GROUNDING_DINO_CONFIG_PATH = "Garage/models/GroundedSegmentAnything/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "Garage/models/checkpoints/GroundedSegmentAnything/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = "Garage/models/checkpoints/GroundedSegmentAnything/sam_vit_h_4b8939.pth"
SAM_ENCODER_VERSION = "vit_h"

class GradioWindow():
    def __init__(self) -> None:
        self.points = []
        self.mask = []
        self.selected_mask = None
        self.segmentation_mask = []
        self.concatenated_masks = None
        self.examples_masks = {
            0: ["dog", "examples/dog_mask.jpg"],
            1: ["bread", "examples/bread_mask.jpg"],
            2: ["room", "examples/room_mask.jpg"],
            3: ["spoon", "examples/spoon_mask.jpg"],
            4: ["cat", "examples/image_mask.jpg"], 
        }

        self.GROUNDING_DINO_CONFIG_PATH = GROUNDING_DINO_CONFIG_PATH
        self.GROUNDING_DINO_CHECKPOINT_PATH = GROUNDING_DINO_CHECKPOINT_PATH
        self.model_type = SAM_ENCODER_VERSION
        self.SAM_CHECKPOINT_PATH = SAM_CHECKPOINT_PATH

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # for debug
        # self.augmenter = None
        self.augmenter = Augmenter(device=self.device)
        self.setup_model()
        self.main()

    def main(self):
        with gr.Blocks() as self.demo:
            with gr.Row():
                input_img = gr.Image(type="pil", label="Input image", interactive=True)
                selected_mask = gr.Image(type="pil", label="Selected Mask", interactive=True)
                segmented_img = gr.Image(type="pil", label="Selected Segment")

            with gr.Row():
                with gr.Group():
                    gr.Markdown(
                        "## Grounded Segmentation\n"
                        "#### This tool segments the object in the image based on the text prompt via GroundedSAM model. "
                        "You can also load the mask of the object to segment or choose one of the examples below.\n"
                    )
                    self.current_object = gr.Textbox(label="Current object")
                    with gr.Accordion("Advanced options", open=False):
                        self.use_mask = gr.Checkbox(label="Use segmentation mask", value=False)
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
                        [self.examples_masks[0][1]],
                        [self.examples_masks[1][1]],
                        [self.examples_masks[2][1]],
                        [self.examples_masks[3][1]],
                        [self.examples_masks[4][1]], 
                        ], 
                        inputs=[selected_mask, input_img],    
                        outputs=[segmented_img, self.current_object, self.use_mask],
                        fn=self.set_mask,
                        run_on_click=True
                    )

            with gr.Row():
                with gr.Column(): 
                    with gr.Group():
                        gr.Markdown(
                        "## Augmentation\n"
                        "#### This tool generates an augmented image based on the input image, the object to augment, and the target object. "
                        "If you don't specify the target object, the model will generate a random object. "
                        "You can also specify the number of steps, guidance scale, and seed for the generation process.\n"
                        )
                        self.target_object = gr.Textbox(label="Target object")

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
            selected_mask.upload(
                self.set_mask,
                inputs=[selected_mask, input_img],    
                outputs=[segmented_img, self.current_object, self.use_mask],
            )

            segment_object.click(
                self.detect,
                inputs=[input_img, self.current_object, 
                        self.use_mask, box_threshold, 
                        text_threshold],
                outputs=[segmented_img, selected_mask]
            )

            self.use_mask.change(
                fn=self.change_mask_type,
                inputs=[input_img, self.use_mask],
                outputs=[selected_mask, segmented_img],
            )

            segmented_img.select(
                self.select_mask,
                inputs=[input_img],
                outputs=[selected_mask, segmented_img],
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

    def change_mask_type(self, image, is_segmmask):
        self.selected_mask = None
        masks = []
        self.mask = []
        if is_segmmask:
            for segm_mask in self.segmentation_mask:
                gray_mask = np.array(segm_mask)
                if gray_mask.ndim == 3:
                    gray_mask  = gray_mask[:, :, 0]
                    gray_mask = np.where(gray_mask > 200, True, False)
                masks.append(gray_mask)
                self.mask.append(Image.fromarray(gray_mask))
            res, common_mask = self.concatenate_masks(masks, image)
        else:
            for segm_mask in self.segmentation_mask:
                mask = self.get_bbox_mask(segm_mask)
                gray_mask = np.array(mask)
                masks.append(gray_mask)
                self.mask.append(Image.fromarray(gray_mask))
            res, common_mask = self.concatenate_masks(masks, image)
        return common_mask, res

    def get_bbox_mask(self, mask):
        bbox = mask.getbbox()
        new_mask = Image.new("L", mask.size, 0)  # Start with an all-black mask
        draw = ImageDraw.Draw(new_mask)
        if bbox:
            draw.rectangle(bbox, fill=255)
        return new_mask    

    def select_mask(self, image: Image, evt: gr.SelectData):
        self.points = [evt.index[0], evt.index[1]]
        selected_mask = np.zeros_like(image)
        self.selected_mask = None
        for mask in self.mask:
            mask = np.array(mask)
            plt.imshow(mask)
            plt.show()
            print(f"SELECT MASK {mask.shape}, unique {np.unique(mask)}")
            if mask[self.points[1]][self.points[0]]:
                self.selected_mask = Image.fromarray(mask)
                color = np.array([30 / 255, 144 / 255, 255 / 255])
                selected_mask[mask > 0] = color.reshape(1, 1, -1) * 255
                selected_mask = Image.fromarray(selected_mask, mode="RGB")
                break

        res = self.show_mask(selected_mask, image)
        self.concatenated_masks = res
        return self.selected_mask, res
    
    def set_mask(self, mask: Image, image: Image):
        self.selected_mask = mask
        self.segmentation_mask = [mask]
        current_object = None

        for key, value in self.examples_masks.items():
            m = Image.open(value[1])
            if np.array_equal(np.array(m), np.array(mask)):
                current_object = value[0]
                break

        gray_mask = np.array(mask)
        gray_mask  = gray_mask[:, :, 0]
        bin_mask = np.where(gray_mask > 200, True, False)
        print(f"SET MASK {bin_mask.shape}, unique {np.unique(bin_mask)}")

        _, common_mask = self.concatenate_masks([bin_mask], image)
        self.mask = [Image.fromarray(bin_mask)]
        res = self.show_mask(common_mask, image)
        self.concatenated_masks = res
        return res, current_object, True

    def detect(self, image: Image, prompt: str, is_segmmask: bool, 
               box_threshold: float, text_threshold: float):
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

        if len(detections.mask) == 0:
            return np.array(image), Image.fromarray(np.zeros_like(np.array(image)))
        
        self.segmentation_mask = []
        for mask in detections.mask:    
            self.segmentation_mask.append(Image.fromarray(mask))

        if is_segmmask:
            image, common_mask = self.concatenate_masks(detections.mask, image)
        else:
            masks = []
            for mask in detections.mask:
                bbox_mask = self.get_bbox_mask(Image.fromarray(mask))
                masks.append(np.array(bbox_mask))
            image, common_mask = self.concatenate_masks(masks, image)

        return image, common_mask
    
    def concatenate_masks(self, masks: np.ndarray, image: Image) -> np.ndarray:
        self.mask = []
        random_color = False
        common_mask = np.zeros_like(image)
        for i, mask in enumerate(masks):
            if random_color:
                color = np.concatenate([np.random.random(3)], axis=0)
            else:
                color = np.array([30 / 255, 144 / 255, 255 / 255])
            
            self.mask.append(Image.fromarray(mask))
            common_mask[mask > 0] = color.reshape(1, 1, -1) * 255
            random_color = True
        
        common_mask = Image.fromarray(common_mask, mode="RGB")
        image = self.show_mask(common_mask, image, random_color)

        common_mask = np.where(np.array(common_mask) != 0, 255, 0).astype(np.uint8)
        return Image.fromarray(image), Image.fromarray(common_mask)
    
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
        mask, image = np.array(mask), np.array(image)
        target_size = (image.shape[1], image.shape[0])  # width, height
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
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

    def augment_image(self, image: Image, 
                      current_object: str, new_objects_list: str,
                      ddim_steps: int, guidance_scale: int, seed: int, return_prompt: str) -> tuple:
        
        if self.selected_mask:
            mask = self.selected_mask
        else:
            mask = self.mask[np.random.choice(len(self.mask))]

        new_objects_list = new_objects_list.split(", ")

        result, (prompt, _) = self.augmenter(
        image=image,
        mask=mask,
        current_object=current_object,
        new_objects_list=new_objects_list,
        ddim_steps=ddim_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        return_prompt=return_prompt
        )

        # # for debug
        # result = mask
        # prompt = "just mask" 
        
        if not return_prompt:
            prompt = ""

        prompt_message = f"<div class=\"message\" style=\"text-align: center; \
                                font-size: 18px;\">Generated prompt: {prompt}</div>"
        return result, prompt_message
    
    
if __name__ == "__main__":
    window = GradioWindow()
    window.demo.launch(share=False)
    window.demo.close()