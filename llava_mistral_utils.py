import os

def get_output_mistral(model, prompt):
    answer = model.infer_prompt(prompt)
    answer = answer[answer.rfind('ASSISTANT:') + 10:]
    return ' '.join(answer.split())


def get_output_llava(model, processor, prompt, image, device):
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    for elem in inputs:
        inputs[elem] = inputs[elem].to(device)

    generate_ids = model.generate(**inputs, max_new_tokens=50)
    answer = processor.batch_decode(generate_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)[0]
    answer = answer[answer.rfind('ASSISTANT:') + 10:]
    return answer


def generate_prompt_llava_mistral(pil_image, 
                                 current_object,  
                                 llava_model, 
                                 llava_processor, 
                                 mistral_model, 
                                 new_objects_list=None,
                                 prompt_path=None,
                                 save_prompt=False, 
                                 device='cuda:0'):
    """
    new_object: either str with the name of the new object or None, then the name generates mistral
    """
    PROMPT0 = "USER: <image>\nI gave you an image. What do you see there? Give me an answer in two or three sequences. ASSISTANT: "
    image_description = get_output_llava(llava_model, llava_processor, PROMPT0, pil_image, device)
    
    PROMPT1v1 = "USER: Imagine you are a object replacer. Your task is generating a replacement object instead of the existing object on the "\
                "scene. It's important that the new object is not the same as the existing one. I will give you a description of the scene and "\
                "the existing object. You must give me an object which could be "\
                f"depicted instead of existing object. So, image description: {image_description}, existing object: {current_object}. You should return "\
                "only a name of new object and nothing else. ASSISTANT: a"

    PROMPT1v2 = "USER: Imagine you are a object replacer. Your task is generating a replacement object instead of the existing object on the "\
                "scene. It's important that the new object is not the same as the existing one. "\
                "I will give you a description of the scene, existing object and a list of potential new objects. "\
                f"So, image description: {image_description}, existing "\
                f"object: {current_object}, a list of potential objects.: {new_objects_list}. You should select "\
                "and return only the name of the new object from the provided list, which is different from the existing object. ASSISTANT: "

    new_object = get_output_mistral(mistral_model, PROMPT1v1 if new_objects_list is None else PROMPT1v2).lower()

    PROMPT2 = "USER: Imagine you are describing the visual appearance of an object using only adjectives. "\
            "Your task is to provide a brief and detailed visual description using only adjectives that convey the object's appearance."\
            "Description includes the object's color, and any other relevant visual details. Do not include any extra words or synonyms for the object's name. "\
            "Then, at the end of the description, include the name of the object. "\
            f"So, оbject: {new_object}. ASSISTANT: a"

    prompt = get_output_mistral(mistral_model, PROMPT2)
    
    if save_prompt:
        file_path = prompt_path
        # Save the prompt to the file
        with open(file_path, 'w') as f:
            f.write(prompt)

    return new_object, prompt