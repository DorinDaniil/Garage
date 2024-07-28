import cv2
import numpy as np
import torch
import time
from PIL import Image
import gradio as gr
from GenerativeAugmentations.models.GroundedSegmentAnything.segment_anything.segment_anything import SamPredictor, sam_model_registry
from GenerativeAugmentations import Augmenter

MODEL_DICT = dict(
    vit_h="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",  # yapf: disable  # noqa
    vit_l="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",  # yapf: disable  # noqa
    vit_b="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",  # yapf: disable  # noqa
)


class GradioWindow():
    def __init__(self) -> None:
        self.saved_points = []
        self.saved_labels = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "vit_b"

        self.augmenter = None
        # self.augmenter = Augmenter(device=self.device)
        self.predictor = self.setup_model()
        self.main()

    def main(self):
        with gr.Blocks() as self.demo:
            with gr.Row():
                input_img = gr.Image(label="Input", interactive=True)
                segmented_img = gr.Image(label="Selected Segment")

            with gr.Row():
                with gr.Column(): 
                    loading_status = gr.Markdown(
                        "<div class=\"message\" style=\"text-align: center; \
                            font-size: 24px;\">Please load image</div>", 
                        visible=True)
                    mask_level = gr.Slider(
                        minimum=0,
                        maximum=2,
                        value=1,
                        step=1,
                        label="Masking level",
                        info="(Whole - Part - Subpart) level",
                    )
                    is_positive_box = gr.Checkbox(value=True, label="Positive point")
                    self.current_object = gr.Textbox(label="Current object", value="cat")
                    self.target_object = gr.Textbox(label="Target object", value="dog")

                    self.iter_number = gr.Number(value=50, label="Steps")
                    self.guidance_scale = gr.Number(value=5, label="Guidance Scale")
                    self.seed = gr.Number(value=1, label="Seed")

                    enter_prompt = gr.Button("Augment Image")
                    
                    reset = gr.Button("Reset Points")

                with gr.Column():
                    augmented_img = gr.Image(label="Augmented Image")

            # Connect the UI and logic
            input_img.upload(
                fn=self.set_image, 
                inputs=[input_img],
                outputs=[loading_status],
                )

            input_img.select(
                self.segment_anything,
                inputs=[input_img, mask_level, is_positive_box],
                outputs=[segmented_img],
            )

            enter_prompt.click(
                self.augment_image,
                inputs=[input_img, self.current_object, self.target_object, 
                        self.iter_number, self.guidance_scale, self.seed],
                outputs=[augmented_img],
            )

            reset.click(self.reset_points)

    def setup_model(self) -> SamPredictor:
        sam = sam_model_registry[self.model_type]()
        sam.load_state_dict(torch.utils.model_zoo.load_url(MODEL_DICT[self.model_type]))
        sam.to(device=self.device)

        return SamPredictor(sam)
    
    def set_image(self, img) -> None:
        """Set the image for the predictor."""
        self.predictor.set_image(img)
        print("Image loaded!")
        return "<div class=\"message\" style=\"text-align: center; font-size: 24px;\">Image Loaded!</div>"
        
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

    def segment_anything(self, img, mask_level: int, is_positive: bool, evt: gr.SelectData):
        """Segment the selected region."""
        global masks
        mask_level = 2 - mask_level
        self.saved_points.append([evt.index[0], evt.index[1]])
        self.saved_labels.append(1 if is_positive else 0)
        input_point = np.array(self.saved_points)
        input_label = np.array(self.saved_labels)

        # Predict the mask
        # with torch.amp.autocast(device_type=str(self.device)):
        
        masks, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # mask has a shape of [3, h, w]
        self.masks = masks[mask_level : mask_level + 1, ...]
        print(self.masks.shape, type(self.masks), np.unique(self.masks))

        res = self.show_mask(self.masks, img)
        res = self.show_points(input_point, input_label, res)
        return res

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