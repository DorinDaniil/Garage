import random
import argparse
import os
import cv2
import json
import numpy as np
import torch
from accelerate import Accelerator
from controlnet_aux import HEDdetector, OpenposeDetector
from PIL import Image, ImageFilter
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from pycocotools import mask as mask_utils
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from pipeline.pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from pipeline.pipeline_PowerPaint_ControlNet import StableDiffusionControlNetInpaintPipeline as controlnetPipeline
from power_paint_utils import TokenizerWrapper, add_tokens
from safetensors.torch import load_model


NEGATIVE_PROMPT = "text, bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, "\
                    "duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, "\
                    "low quality, low res, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def poisson_blend(
    orig_img: np.ndarray,
    fake_img: np.ndarray,
    mask: np.ndarray,
    pad_width: int = 32,
    dilation: int = 33
) -> np.ndarray:
    """Does poisson blending with some tricks.

    Args:
        orig_img (np.ndarray): Original image.
        fake_img (np.ndarray): Generated fake image to blend.
        mask (np.ndarray): Binary 0-1 mask to use for blending.
        pad_width (np.ndarray): Amount of padding to add before blending (useful to avoid some issues).
        dilation (np.ndarray): Amount of dilation to add to the mask before blending (useful to avoid some issues).

    Returns:
        np.ndarray: Blended image.
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    padding_config = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))
    padded_fake_img = np.pad(fake_img, pad_width=padding_config, mode="reflect")
    padded_orig_img = np.pad(orig_img, pad_width=padding_config, mode="reflect")
    padded_orig_img[:pad_width, :, :] = padded_fake_img[:pad_width, :, :]
    padded_orig_img[:, :pad_width, :] = padded_fake_img[:, :pad_width, :]
    padded_orig_img[-pad_width:, :, :] = padded_fake_img[-pad_width:, :, :]
    padded_orig_img[:, -pad_width:, :] = padded_fake_img[:, -pad_width:, :]
    padded_mask = np.pad(mask, pad_width=padding_config[:2], mode="reflect").astype(np.uint8)
    # padded_dmask = cv2.dilate(padded_mask, np.ones((dilation, dilation), np.uint8), iterations=1)
    x_min, y_min, rect_w, rect_h = cv2.boundingRect(padded_mask)
    center = (x_min + rect_w // 2, y_min + rect_h // 2)
    output = cv2.seamlessClone(padded_fake_img, padded_orig_img, padded_mask, center, cv2.NORMAL_CLONE)
    output = output[pad_width:-pad_width, pad_width:-pad_width]
    return output


def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad(), torch.autocast(DEVICE):
        depth_map = depth_estimator(image).predicted_depth

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


def add_task(prompt, negative_prompt, control_type):
    # print(control_type)
    if control_type == "object-removal":
        promptA = "empty scene blur " + prompt + " P_ctxt"
        promptB = "empty scene blur " + prompt + " P_ctxt"
        negative_promptA = negative_prompt + " P_obj"
        negative_promptB = negative_prompt + " P_obj"
    elif control_type == "shape-guided":
        promptA = prompt + " P_shape"
        promptB = prompt + " P_ctxt"
        negative_promptA = (
            negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry P_shape"
        )
        negative_promptB = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry P_ctxt"
    elif control_type == "image-outpainting":
        promptA = "empty scene " + prompt + " P_ctxt"
        promptB = "empty scene " + prompt + " P_ctxt"
        negative_promptA = negative_prompt + " P_obj"
        negative_promptB = negative_prompt + " P_obj"
    else:
        promptA = prompt + " P_obj"
        promptB = prompt + " P_obj"
        negative_promptA = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry, P_obj"
        negative_promptB = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry, P_obj"

    return promptA, promptB, negative_promptA, negative_promptB


def predict(
    pipe,
    input_image,
    prompt,
    fitting_degree,
    ddim_steps,
    scale,
    seed,
    negative_prompt,
    task,
    vertical_expansion_ratio,
    horizontal_expansion_ratio,
    height=-1,
    width=-1
):
    size1, size2 = input_image["image"].convert("RGB").size

    if task != "image-outpainting":
        if size1 < size2:
            input_image["image"] = input_image["image"].convert("RGB").resize((640, int(size2 / size1 * 640)))
        else:
            input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 640), 640))
    else:
        if size1 < size2:
            input_image["image"] = input_image["image"].convert("RGB").resize((512, int(size2 / size1 * 512)))
        else:
            input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 512), 512))

    if vertical_expansion_ratio != None and horizontal_expansion_ratio != None:
        o_W, o_H = input_image["image"].convert("RGB").size
        c_W = int(horizontal_expansion_ratio * o_W)
        c_H = int(vertical_expansion_ratio * o_H)

        expand_img = np.ones((c_H, c_W, 3), dtype=np.uint8) * 127
        original_img = np.array(input_image["image"])
        expand_img[
            int((c_H - o_H) / 2.0) : int((c_H - o_H) / 2.0) + o_H,
            int((c_W - o_W) / 2.0) : int((c_W - o_W) / 2.0) + o_W,
            :,
        ] = original_img

        blurry_gap = 10

        expand_mask = np.ones((c_H, c_W, 3), dtype=np.uint8) * 255
        if vertical_expansion_ratio == 1 and horizontal_expansion_ratio != 1:
            expand_mask[
                int((c_H - o_H) / 2.0) : int((c_H - o_H) / 2.0) + o_H,
                int((c_W - o_W) / 2.0) + blurry_gap : int((c_W - o_W) / 2.0) + o_W - blurry_gap,
                :,
            ] = 0
        elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio != 1:
            expand_mask[
                int((c_H - o_H) / 2.0) + blurry_gap : int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                int((c_W - o_W) / 2.0) + blurry_gap : int((c_W - o_W) / 2.0) + o_W - blurry_gap,
                :,
            ] = 0
        elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio == 1:
            expand_mask[
                int((c_H - o_H) / 2.0) + blurry_gap : int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                int((c_W - o_W) / 2.0) : int((c_W - o_W) / 2.0) + o_W,
                :,
            ] = 0

        input_image["image"] = Image.fromarray(expand_img)
        input_image["mask"] = Image.fromarray(expand_mask)

    promptA, promptB, negative_promptA, negative_promptB = add_task(prompt, negative_prompt, task)
    print(promptA, promptB, negative_promptA, negative_promptB)
    img = np.array(input_image["image"].convert("RGB"))
    if height == -1 and width == -1:
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)

        print(H)
    else:
        H = height
        W = width

    input_image["image"] = input_image["image"].resize((H, W))
    input_image["mask"] = input_image["mask"].resize((H, W))
    set_seed(seed)
    result = pipe(
        promptA=promptA,
        promptB=promptB,
        tradoff=fitting_degree,
        tradoff_nag=fitting_degree,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        image=input_image["image"].convert("RGB"),
        mask_image=input_image["mask"].convert("RGB"),
        width=H,
        height=W,
        guidance_scale=scale,
        num_inference_steps=ddim_steps,
    ).images[0]
    mask_np = np.array(input_image["mask"].convert("RGB"))
    red = np.array(result).astype("float") * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray(
        (
            result_m.astype("float") * (1 - mask_np.astype("float") / 512.0) + mask_np.astype("float") / 512.0 * red
        ).astype("uint8")
    )
    m_img = input_image["mask"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=3))
    m_img = np.asarray(m_img) / 255.0
    img_np = np.asarray(input_image["image"].convert("RGB")) / 255.0
    ours_np = np.asarray(result) / 255.0
    ours_np = ours_np * m_img + (1 - m_img) * img_np
    result_paste = Image.fromarray(np.uint8(ours_np * 255))

    dict_res = [input_image["mask"].convert("RGB"), result_m]
    dict_out = [input_image["mask"].convert("RGB"), result_paste]

    return dict_out, dict_res


def predict_controlnet(
    pipe,
    input_image,
    input_control_image,
    control_type,
    prompt,
    ddim_steps,
    scale,
    seed,
    negative_prompt,
    controlnet_conditioning_scale,
):
    promptA = prompt + " P_obj"
    promptB = prompt + " P_obj"
    negative_promptA = negative_prompt
    negative_promptB = negative_prompt
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

    global current_control

    base_control = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype)
    control_pipe = controlnetPipeline(
        pipe.vae, pipe.text_encoder, pipe.tokenizer, pipe.unet, base_control, pipe.scheduler, None, None, False
    )
    control_pipe = control_pipe.to("cuda")
    current_control = "canny"
    if current_control != control_type:
        if control_type == "canny" or control_type is None:
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype
            )
        elif control_type == "pose":
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose", torch_dtype=weight_dtype
            )
        elif control_type == "depth":
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth", torch_dtype=weight_dtype
            )
        else:
            control_pipe.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-hed", torch_dtype=weight_dtype
            )
        control_pipe = control_pipe.to("cuda")
        current_control = control_type

    controlnet_image = input_control_image
    if current_control == "canny":
        controlnet_image = controlnet_image.resize((H, W))
        controlnet_image = np.array(controlnet_image)
        controlnet_image = cv2.Canny(controlnet_image, 100, 200)
        controlnet_image = controlnet_image[:, :, None]
        controlnet_image = np.concatenate([controlnet_image, controlnet_image, controlnet_image], axis=2)
        controlnet_image = Image.fromarray(controlnet_image)
    elif current_control == "pose":
        controlnet_image = openpose(controlnet_image)
    elif current_control == "depth":
        controlnet_image = controlnet_image.resize((H, W))
        controlnet_image = get_depth_map(controlnet_image)
    else:
        controlnet_image = hed(controlnet_image)

    mask_np = np.array(input_image["mask"].convert("RGB"))
    controlnet_image = controlnet_image.resize((H, W))
    set_seed(seed)
    result = control_pipe(
        promptA=promptB,
        promptB=promptA,
        tradoff=1.0,
        tradoff_nag=1.0,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        image=input_image["image"].convert("RGB"),
        mask_image=input_image["mask"].convert("RGB"),
        control_image=controlnet_image,
        width=H,
        height=W,
        guidance_scale=scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=ddim_steps,
    ).images[0]
    red = np.array(result).astype("float") * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(result)
    result_m = Image.fromarray(
        (
            result_m.astype("float") * (1 - mask_np.astype("float") / 512.0) + mask_np.astype("float") / 512.0 * red
        ).astype("uint8")
    )

    mask_np = np.array(input_image["mask"].convert("RGB"))
    m_img = input_image["mask"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=4))
    m_img = np.asarray(m_img) / 255.0
    img_np = np.asarray(input_image["image"].convert("RGB")) / 255.0
    ours_np = np.asarray(result) / 255.0
    ours_np = ours_np * m_img + (1 - m_img) * img_np
    result_paste = Image.fromarray(np.uint8(ours_np * 255))
    return [input_image["image"].convert("RGB"), result_paste], [controlnet_image, result_m]


def infer(
    pipe,
    input_image,
    text_guided_prompt,
    text_guided_negative_prompt,
    shape_guided_prompt,
    shape_guided_negative_prompt,
    fitting_degree,
    ddim_steps,
    scale,
    seed,
    task,
    enable_control,
    input_control_image,
    control_type,
    vertical_expansion_ratio,
    horizontal_expansion_ratio,
    outpaint_prompt,
    outpaint_negative_prompt,
    controlnet_conditioning_scale,
    removal_prompt,
    removal_negative_prompt,
):
    if task == "text-guided":
        prompt = text_guided_prompt
        negative_prompt = text_guided_negative_prompt
    elif task == "shape-guided":
        prompt = shape_guided_prompt
        negative_prompt = shape_guided_negative_prompt
    elif task == "object-removal":
        prompt = removal_prompt
        negative_prompt = removal_negative_prompt
    elif task == "image-outpainting":
        prompt = outpaint_prompt
        negative_prompt = outpaint_negative_prompt
        return predict(
            pipe,
            input_image,
            prompt,
            fitting_degree,
            ddim_steps,
            scale,
            seed,
            negative_prompt,
            task,
            vertical_expansion_ratio,
            horizontal_expansion_ratio,
        )
    else:
        task = "text-guided"
        prompt = text_guided_prompt
        negative_prompt = text_guided_negative_prompt

    return predict(
                pipe,
                input_image,
                prompt,
                fitting_degree,
                ddim_steps, scale,
                seed,
                negative_prompt,
                task,
                None,
                None
    )


def dilate_mask(mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def mask_bbox(mask, bbox):
    for x in range(bbox[0], bbox[0] + bbox[2] + 1):
        for y in range(bbox[1], bbox[1] + bbox[3] + 1):
            mask[y, x] = 1
    return mask


def get_cropped(image, mask_info, mask):
    bbox = mask_info['bbox']
    h, w = mask_info['segmentation']['size']
    padding = int(min(h, w) * 0.05)
    left_top_x, left_top_y = max(0, bbox[0] - padding), max(0, bbox[1] - padding)
    right_bottom_x, right_bottom_y = min(w, left_top_x + bbox[2] + 2*padding), min(h, left_top_y + bbox[3] + 2*padding)
    new_W, new_H = right_bottom_x - left_top_x, right_bottom_y - left_top_y
    new_image = image.crop((left_top_x, left_top_y, right_bottom_x, right_bottom_y))
    new_mask = Image.fromarray(mask.astype(bool)).crop((left_top_x, left_top_y, right_bottom_x, right_bottom_y))

    k = int(min(512 / new_H, 512 / new_W)) + 1
    new_image = new_image.resize((k * new_W, k * new_H))
    new_mask = new_mask.resize((k * new_W, k * new_H))

    return new_image, new_mask, (left_top_x, left_top_y), (new_W, new_H)


def preprocess_image(image, image_info, height, width):
    min_side = min(height, width)

    masks = []
    masks_infos = []
    for mask_info in image_info['annots']:
        mask = mask_info['segmentation']
        mask = mask_utils.decode(mask)
        if min_side > 1024:
            if min_side == height:
                k = 1024 / height
            else:
                k = 1024 / width

            reduced_h, reduced_w = int(k * height), int(k * width)
            mask_info['segmentation']['size'] = [int(k * height), int(k * width)]
            mask_info['area'] *= k**2
            mask_info['bbox'] = np.array(np.array(mask_info['bbox']) * k).astype(int)
            mask = Image.fromarray(mask.astype(bool))
            image = image.resize((reduced_w, reduced_h))
            mask = np.array(mask.resize((reduced_w, reduced_h)), dtype=np.uint8)
        masks.append(mask.copy())
        masks_infos.append(mask_info.copy())

    return image, masks, masks_infos


def get_image(path, metainfo, image_id, dilate=True, crop=True):

    image = Image.open(path + '/' + metainfo[image_id]['file_name'])
    image_info = metainfo[image_id]

    height, width = image_info['height'], image_info['width']

    preprocessed_image, masks, masks_infos = preprocess_image(image, image_info, height, width)
    dilate_kernel = int(min(preprocessed_image.size) * 0.03)

    assert len(masks) == len(masks_infos), "Lengths of 'masks' and 'masks_infos' do not match!\n"

    images = []
    masks_modified = []
    objects_names = []
    objects_ids = []
    tasks = []
    new_objects = []
    crops = []

    for mask_, mask_info_ in zip(masks, masks_infos):
        mask = mask_.copy()
        mask_info = mask_info_.copy()

        if mask_info['task'] == 'object-removal':
            crop = False

        new_object = mask_info['new_object']

        if dilate:
            mask = dilate_mask(mask, dilate_kernel)

        if crop:
            image, mask, left_top, shape = get_cropped(preprocessed_image, mask_info, mask)

        object_name = mask_info['category_name']
        object_id = mask_info['id']

        images.append(image)
        masks_modified.append(mask)
        objects_names.append(object_name)
        objects_ids.append(object_id)
        tasks.append(mask_info['task'])
        new_objects.append(new_object)
        crops.append(crop if not crop else (left_top, shape))

    return {
        "image": images,
        "source_image": preprocessed_image,
        "mask": masks_modified,
        "object": objects_names,
        "id": objects_ids,
        "task": tasks,
        "new_object": new_objects,
        "crop": crops
    }


def parse_task_params(image):
    text_guided_prompt = ""
    text_guided_negative_prompt = NEGATIVE_PROMPT
    shape_guided_prompt = ""
    shape_guided_negative_prompt = NEGATIVE_PROMPT
    fitting_degree = 1
    ddim_steps = 50
    scale = 15
    seed = random.randint(0, 1 << 32)
    task = image["task"]
    enable_control = True
    input_control_image = None
    control_type = None
    vertical_expansion_ratio = 1
    horizontal_expansion_ratio = 1
    outpaint_prompt = ""
    outpaint_negative_prompt = NEGATIVE_PROMPT
    controlnet_conditioning_scale = 0.5
    removal_prompt = ""
    removal_negative_prompt = NEGATIVE_PROMPT

    if image["task"] == "text-guided":
        text_guided_prompt = image["new_object"]
        scale = 12
        control_type = "depth"
    elif image["task"] == "shape-guided":
        shape_guided_prompt = image["new_object"]
    elif image["task"] == "object-removal":
        removal_negative_prompt = image["object"]
        scale = 20
    elif task == "image-outpainting":
        raise NotImplementedError

    return (
        text_guided_prompt,
        text_guided_negative_prompt,
        shape_guided_prompt,
        shape_guided_negative_prompt,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        task,
        enable_control,
        input_control_image,
        control_type,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
        outpaint_prompt,
        outpaint_negative_prompt,
        controlnet_conditioning_scale,
        removal_prompt,
        removal_negative_prompt
    )


def prepare_pipe():
    weight_dtype = torch.float16

    pipe = Pipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=weight_dtype)

    load_model(pipe.unet, "./models/unet/unet.safetensors")
    load_model(pipe.text_encoder, "./models/unet/text_encoder.safetensors")
    return pipe


def main(args):
    accelerator = Accelerator()

    torch.set_grad_enabled(False)
    global weight_dtype
    weight_dtype = torch.float16

    pipe = Pipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=weight_dtype)
    pipe.tokenizer = TokenizerWrapper(
        from_pretrained="runwayml/stable-diffusion-v1-5", subfolder="tokenizer", revision=None
    )

    add_tokens(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
        initialize_tokens=["a", "a", "a"],
        num_vectors_per_token=10,
    )

    load_model(pipe.unet, "./models/unet/unet.safetensors")
    load_model(pipe.text_encoder, "./models/unet/text_encoder.safetensors")
    pipe = pipe.to(accelerator.device)

    with accelerator.split_between_processes(file_idxs) as chunked_files:
        for file_id in chunked_files:
            image = get_image(args.images_path, metainfo, file_id)

            path = os.path.join(args.output_path, file_id)
            os.mkdir(path)

            image["source_image"].save(path + '/' + 'source.png')
            for image_, mask_, obj_, obj_id_, task_, new_obj_, crop_ in zip(
                image["image"], image["mask"], image["object"], image["id"], image["task"], image["new_object"], image["crop"]
            ):
                image_one = {
                    "image": image_,
                    "source_image": image["source_image"],
                    "mask": mask_,
                    "object": obj_,
                    "id": obj_id_,
                    "task": task_,
                    "new_object": new_obj_,
                    "crop": crop_
                }

                inpaint_result, gallery = infer(
                    pipe,
                    image_one,
                    *parse_task_params(image_one)
                )
                print(inpaint_result)
                source = image_one['source_image'].copy()

                if image_one['crop']:
                    to_paste_image = inpaint_result[1].resize(image_one['crop'][1])

                    orig_path = path + '/' + f"{str(image_one['id'])}_orig.png"
                    fake_path = path + '/' + f"{str(image_one['id'])}_fake.png"
                    mask_path = path + '/' + f"{str(image_one['id'])}_mask.png"

                    image_one['image'].save(orig_path)
                    inpaint_result[1].save(fake_path)
                    image_one['mask'].save(mask_path)

                    orig = cv2.imread(orig_path)
                    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

                    fake = cv2.imread(fake_path)
                    fake = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)

                    mask = cv2.imread(mask_path)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                    result = poisson_blend(
                        orig,
                        fake,
                        mask
                    )
                    result = Image.fromarray(np.uint8(result)).resize(image_one['crop'][1])
                    result.save(path + '/' + f"{str(image_one['id'])}_res.png")
                    # result = cv2.merge(result_stack)
                    source.paste(result, image_one['crop'][0])
                    result = source
                else:
                    result = inpaint_result[1]
                # to_paste_image.save(path + '/' + f"{str(image_one['id'])}_paste.png")
                result.save(path + '/' + f"{str(image_one['id'])}.png")
                # gallery[0].resize(image_one['crop'][1]).save(path + '/' + f"{str(image_one['id'])}_mask.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    main(args)
