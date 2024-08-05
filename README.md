# GenerativeAugmentations

Welcome to GenerativeAugmentations, a cutting-edge Python library designed for generative image augmentation! Our library includes the Augmenter model, which leverages advanced machine learning techniques to generate new objects and prompts, seamlessly replacing specified objects in images with new ones using PowerPaint.

## Get Started

```bash
# Clone the Repository
git clone https://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@github.com/DorinDaniil/augmenter_pipeline.git

# Create Virtual Environment with Conda
conda create --name genaug python=3.12
conda activate genaug

# Install Dependencies
pip install -r requirements.txt
```
To use the model download the PowerPaint v2-1 weights.
```bash
git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1/ ./checkpoints/ppt-v2-1
```

To use GroundedSAM follow the installation instructions [here](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main)

```
cd GenerativeAugmentations/models
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything GroundedSegmentAnything
```

### Use Docker
To start with Docker run following commands:
```
# build image
docker build -t augmenter-app .

# run container
docker run -it -d --gpus all -p 7860:7860 --name augmenter augmenter-app

# get into running container
docker exec -u 0 -it augmenter /bin/bash 
cd augmenter_pipeline 
```

To run demo app use followung command:
```
python app.py
```

Don't forget to login in **huggingface**!
```
huggingface-cli login
```

## Usage

Here's a step-by-step guide on how to use the GenerativeAugmentations library to perform image augmentation:

### Import the necessary modules

```python
from GenerativeAugmentations import Augmenter
from PIL import Image
```

### Initialize the Augmenter class

The Augmenter class is the main interface for performing image augmentation. You can initialize it with the device you want to use for computations, which defaults to "cuda".

```python
augmenter = Augmenter(device="cuda")
```

### Prepare your inputs

You will need to provide the following inputs to the Augmenter:

- `image`: The input image in PIL format.
- `mask`: The mask of the object to replace in PIL format.
- `current_object`: The name of the object to be replaced.
- `new_objects_list` (optional): A list of potential new objects. If None, the method will generate a new object.
- `ddim_steps` (optional): The number of denoising steps. More steps mean a slower but potentially higher quality result. Defaults to 50.
- `guidance_scale` (optional): The scale for classifier-free guidance. Higher values lead to results that are more closely linked to the text prompt. Defaults to 5.
- `seed` (optional): Integer value that initializes the random number generator for reproducibility. Defaults to 1.
- `return_prompt` (optional): If True, the method also returns the prompt used for generation and the new object. Defaults to False.

### Perform image augmentation

You can perform image augmentation by calling the Augmenter instance with the prepared inputs:

```python
image = Image.open("path/to/your/image.jpg")
mask = Image.open("path/to/your/mask.jpg")

result, (prompt, new_object) = augmenter(
    image=image,
    mask=mask,
    current_object=current_object,
    new_objects_list=new_objects_list,
    ddim_steps=50,
    guidance_scale=5,
    seed=1,
    return_prompt=True
)
```

The `result` variable will contain the modified image, prompt used for generation and the new object, respectively.

### Examples

For examples of how to use the GenerativeAugmentations refer to the example.ipynb

## Download augmented datasets

To download the [VOC2007 augmentations](https://huggingface.co/datasets/danulkin/VOC2007Augs), follow the instructions:
```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="danulkin/VOC2007Augs", repo_type="dataset", local_dir = "./VOC2007Augs")
```

