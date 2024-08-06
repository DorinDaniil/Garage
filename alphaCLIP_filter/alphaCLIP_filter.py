import torch
from AlphaCLIP import alpha_clip
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import argparse


class MetricAlphaCLIPScore():
    """
    A class to compute the CLIPScore metric for evaluating the alignment between A PART OF THE generated image and a text prompt.
    """

    def __init__(self, weights, device="cuda") -> None:
        """
        Initialize a MetricAlphaCLIPScore object with the specified device.

        Args:
            weights (str): The path to the model weights.
            device (str, optional): The device on which the model will run. Defaults to "cuda".
        """
        self.model, self.preprocess = alpha_clip.load("ViT-B/16",
                                                      alpha_vision_ckpt_pth=weights,
                                                      device=device)
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            # change to (336,336) when using ViT-L/14@336px
            transforms.Normalize(0.5, 0.26)
        ])
        self.device = device

    def evaluate(self, generated_image: Image.Image, mask: Image.Image,
                 prompt: str):
        """
        Evaluate the alignment between the provided image and text prompt using the CLIPScore metric.

        Args:
            generated_image (Image.Image): The generated image for evaluation.
            mask (Image.Image): The binary mask on the generated image.
            prompt (str): The text prompt associated with the generated image.

        Returns:
            float: The computed CLIPScore.
        """
        image = generated_image.convert('RGB')
        mask = np.array(mask)
        # get `binary_mask` array (2-dimensional bool matrix)
        if len(mask.shape) == 2: binary_mask = (mask == 255)
        if len(mask.shape) == 3: binary_mask = (mask[:, :, 0] == 255)
        alpha = self.mask_transform((binary_mask * 255).astype(np.uint8))
        alpha = alpha.half().cuda().unsqueeze(dim=0)
        # calculate image and text features
        image = self.preprocess(image).unsqueeze(0).half().to(self.device)
        text = alpha_clip.tokenize([prompt]).to(self.device)
        with torch.no_grad():
            image_features = self.model.visual(image, alpha)
            text_features = self.model.encode_text(text)
        # normalize
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(dim=-1,
                                                           keepdim=True)
        clip_score = torch.matmul(image_features, text_features.T).item()
        return clip_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float,
                        help="Threshold for filtering images")
    parser.add_argument("--images_dir_path", type=str,
                        help="Path to augmented images")
    parser.add_argument("--masks_dir_path", type=str,
                        help="Path to masks")
    parser.add_argument("--texts_dir_path", type=str,
                        help="Path to text prompts")
    parser.add_argument("--weights", type=str,
                        help="Path to text prompts")
    args = parser.parse_args()

    threshold = args.threshold
    texts_directory = args.texts_dir_path
    masks_directory = args.masks_dir_path
    images_directory = args.images_dir_path
    weights = args.weights

    texts_files = sorted(os.listdir(texts_directory))
    masks_files = sorted(os.listdir(masks_directory))
    images_files = sorted(os.listdir(images_directory))

    for i in range(len(texts_files)):

        text_file_path = os.path.join(texts_directory, texts_files[i])
        mask_file_path = os.path.join(masks_directory, masks_files[i])
        image_file_path = os.path.join(images_directory, images_files[i])

        model = MetricAlphaCLIPScore(weights=weights)

        image = Image.open(image_file_path)
        mask = Image.open(mask_file_path)
        text = None
        with open(text_file_path, 'r') as f:
            text = f.read().strip()

        score = model.evaluate(image, mask, text)
        if score <= threshold:
            os.remove(text_file_path)
            os.remove(mask_file_path)
            os.remove(image_file_path)
