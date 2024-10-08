import torch
import torch.nn as nn
from safetensors.torch import load_model
from transformers import AutoImageProcessor, Dinov2Model

class PhysicsModel(nn.Module):
    @classmethod
    def from_pretrain(cls, checkpoint_path: str, freeze: bool = True):
        model = cls()
        load_model(model, checkpoint_path)

        if freeze:
            model.freeze()

        return model


    def __init__(self):
        super().__init__()
        self.model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.classifier = nn.Linear(768 * 2 + 50, 2)
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    
    
    def freeze(self, full_freeze: bool = True):
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        if not full_freeze:
            self.classifier.requires_grad_(True)

    def forward(self, scene_image, object_image, location, label=None):
        scene_embedding = self.model(scene_image).pooler_output
        object_embedding = self.model(object_image).pooler_output

        location_embedding = self.positional_encoding(location)

        embedding = torch.cat([scene_embedding, object_embedding, location_embedding], dim=-1)

        output = self.classifier(embedding)
        return output

    def run_inference(self, scene_image, object_image, location):
        scene_image = torch.tensor(scene_image).permute(2,0,1).unsqueeze(0).to(self.model.device).float()
        scene_image = self.image_processor(scene_image, return_tensors="pt")['pixel_values']
        object_image = torch.tensor(object_image).permute(2,0,1).unsqueeze(0).to(self.model.device).float()
        object_image = self.image_processor(object_image, return_tensors="pt")['pixel_values']
        location = torch.tensor(location).unsqueeze(0).to(self.model.device).float()

        output = self(scene_image, object_image, location).argmax(dim=-1)

        return output[0].cpu().detach().numpy()

    def positional_encoding(self, tensor, num_encoding_functions=12, include_input=True, log_sampling=True) -> torch.Tensor:
        r"""Apply positional encoding to the input.
        
        Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        num_encoding_functions (optional, int): Number of encoding functions used to
            compute a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            computed positional encoding (default: True).
        log_sampling (optional, bool): Sample logarithmically in frequency space, as
            opposed to linearly (default: True).
        
        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """
        # TESTED
        # Trivially, the input tensor is added to the positional encoding.
        encoding = [tensor] if include_input else []
        # Now, encode the input using a set of high-frequency functions and append the
        # resulting values to the encoding.
        frequency_bands = None
        if log_sampling:
          frequency_bands = 2.0 ** torch.linspace(
                0.0,
                num_encoding_functions - 1,
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
          frequency_bands = torch.linspace(
              2.0 ** 0.0,
              2.0 ** (num_encoding_functions - 1),
              num_encoding_functions,
              dtype=tensor.dtype,
              device=tensor.device,
          )
        
        for freq in frequency_bands:
          for func in [torch.sin, torch.cos]:
              encoding.append(func(tensor * freq))
        
        # Special case, for no positional encoding
        if len(encoding) == 1:
          return encoding[0]
        else:
          return torch.cat(encoding, dim=-1)


        
