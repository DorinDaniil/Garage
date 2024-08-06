# Filtering Images Using the AlphaCLIP Model

This script filters images, their corresponding masks, and prompts by deleting those for which the scalar product between the text and image embedding vectors is less than the specified threshold.

The text prompt, image and mask for the corresponding check must be named in the same way.

Weights are downloaded in a bash script.

## Setup Instructions

1. Execute the preprocessing script:
   ```bash
   bash preproc.sh
   ```
2. Run the Python script with the necessary parameters. For example:
   ```bash
   python3 alphaCLIP_filter.py --threshold 0.28 --images_dir_path path/to/imgs --masks_dir_path path/to/masks --texts_dir_path path/to/texts --weights path/to/weights
   ```

## Parameters

- `--threshold`: Set the threshold for filtering. Images with a scalar product below this value will be deleted.
- `--images_dir_path`: Specify the path to the directory containing images.
- `--masks_dir_path`: Specify the path to the directory containing masks.
- `--texts_dir_path`: Specify the path to the directory containing texts.
- `--weights`: Specify the path to the model weights.

## Example of directories structure:

- `images/001.jpg`
- `texts/001.txt`
- `masks/001.jpg`
