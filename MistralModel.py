import torch
import transformers

class MistralModel:
    def __init__(
        self,
        device_str='cuda',
        model_name='Intel/neural-chat-7b-v3-1',
        ):
        
        self.device = torch.device(device_str)
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.generation_params = {
            "do_sample": True,
            "temperature": 1,
            "top_p": 0.90,
            "top_k": 40,
            "max_new_tokens": 256,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
    def infer_prompt(
        self,
        prompt
        ):
        # Tokenize and encode the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(self.device)
        
        # Generate a response
        outputs = self.model.generate(inputs, num_return_sequences=1, **self.generation_params)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        return response
        
    def expand_prompt(
        self,
        prompt,
        max_length=4096
        ):
        system_input = "You are a prompt engineer. Your mission is to expand prompts written by user. You should provide the best prompt "\
                    "for text to image generation in English."
        prompt = f"### System:\n{system_input}\n### User:\n{prompt}\n### Assistant:\n"
        
        response = self.infer_prompt(prompt)
        result = response.split("### Assistant:\n")[-1]
        if len(result) > max_length:
            result = result[:4096]
        return result