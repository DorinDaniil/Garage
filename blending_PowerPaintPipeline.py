import random
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from PowerPaintPipeline import StableDiffusionInpaintPipeline as Pipeline
from PowerPaintPipeline import TokenizerWrapper, add_tokens
from safetensors.torch import load_model


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
    #padded_dmask = cv2.dilate(padded_mask, np.ones((dilation, dilation), np.uint8), iterations=1)
    padded_mask = cv2.GaussianBlur(padded_mask, (0, 0), sigmaX=1)
    x_min, y_min, rect_w, rect_h = cv2.boundingRect(padded_mask)
    center = (x_min + rect_w // 2, y_min + rect_h // 2)
    output = cv2.seamlessClone(padded_fake_img, padded_orig_img, padded_mask, center, cv2.NORMAL_CLONE)
    output = output[pad_width:-pad_width, pad_width:-pad_width]
    return output


def add_task(prompt, negative_prompt, control_type):
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
    img = np.array(input_image["image"].convert("RGB"))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)

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