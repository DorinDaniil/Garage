Overview
--------

The AugmenterPipeline is a model that utilizes Mistral, Llava, and StableDiffusionInpaintPipeline to replace a specified object in a given image with a new one. The class generates a new object and a prompt, and then replaces the specified object in the image with the new one.

Usage
-----

To use the AugmenterPipeline, first create an instance of the class with the desired device for model inference:
```python
augmenter = augmenter(device='cuda')
```
Then, call the `__call__` method of the class to replace an object in an image:
```python
modified_image, new_object, prompt = augmenter(pil_image, pil_mask, current_object, new_objects_list=None, num_inference_steps=200, guidance_scale=6, return_prompt=True)
```
The `__call__` method takes the following parameters:

* `pil_image` (Image.Image): The input image.
* `pil_mask` (Image.Image): The mask of object to replace.
* `current_object` (str): The name of the object to be replaced.
* `new_objects_list` (Optional[List[str]]): A list of potential new objects. If None, the method will generate a new object.
* `num_inference_steps` (int): The number of denoising steps. More steps mean a slower but potentially higher quality result.
* `guidance_scale` (int): The scale for classifier-free guidance. Higher values lead to results that are more closely linked to the text prompt.
* `return_prompt` (bool): If True, the method also returns the new object and the prompt used for generation.

The `__call__` method returns a tuple containing the modified image and, optionally, the new object and the prompt used for generation.

Note: The first time an instance of the AugmenterPipeline class is created, it will automatically download the necessary UNET model weights to the same directory as the model and load them during initialization.

Examples
--------

For examples of how to use the AugmenterPipeline, please refer to the provided Jupyter notebook.
