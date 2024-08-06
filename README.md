# Garage: Generative Augmentation Framework for Transforming Object Representations in Images

![Project Page](examples/AbstractGARAGE.jpeg)

Welcome to **Garage**, a cutting-edge Python library designed for generative image augmentation! Our library includes the Augmenter model, which leverages advanced machine learning techniques to generate new objects and prompts, seamlessly replacing specified objects in images with new ones using PowerPaint.

## Get Started

```bash
# Clone the Repository
git clone https://github.com/DorinDaniil/Garage.git

# Create Virtual Environment with Conda
conda create --name garage python=3.10
conda activate garage

# Install Dependencies
pip install -r requirements.txt
```
To use **Garage** library download the PowerPaint v2-1 weights.
```bash
git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1/ ./checkpoints/ppt-v2-1
```
### Use Demo
You can launch the Gradio interface for **Garage** by running the following command:
```bash
# Download the PowerPaint v2-1 weights
git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1/ ./checkpoints/ppt-v2-1

# Download the Grounded SAM weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

conda activate garage
python app.py --share 
```

### Use Docker
Dockerfile authomatically starts demo app in file [app.py](app.py)

To start with Docker run following commands:
```bash
# build image
docker build -t garage-app .

# run container
docker run -it --gpus all -p 7860:7860 --name garage garage-app
```

To use demo app wait for the app to load and go to the following link in your browser:
```
Running on local URL:  http://0.0.0.0:7860
```

## Usage

Here's a step-by-step guide on how to use the **Garage** library to perform image augmentation:

### Import the necessary modules

```python
from Garage import Augmenter
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
    current_object="replacement object",
    new_objects_list=None,
    ddim_steps=50,
    guidance_scale=5,
    seed=1,
    return_prompt=True
)
```

The `result` variable will contain the modified image, prompt used for generation and the new object, respectively.

### Examples

More examples of how to use **Garage** refer to the `example.ipynb`

## Download augmented datasets

To download the [VOC2007 augmentations](https://huggingface.co/datasets/danulkin/VOC2007Augs), follow the instructions:
```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="danulkin/VOC2007Augs", repo_type="dataset", local_dir = "./VOC2007Augs")
```

## Contact Us
**Andrei Filatov**: filatovandreiv@gmail.com

**Daniil Dorin**: dorin.dd@phystech.edu

**Ulyana Izmesteva** izmesteva.ua@phystech.edu

