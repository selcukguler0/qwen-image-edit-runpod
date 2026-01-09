import torch
from diffusers import QwenImageEditPlusPipeline

model_id = "Qwen/Qwen-Image-Edit-2511"
# Sadece indir, save_pretrained satırını SİL
pipe = QwenImageEditPlusPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
print("Model başarıyla indirildi.")
