import torch
from diffusers import QwenImageEditPlusPipeline

# Modeli belirtilen dizine indir
model_id = "Qwen/Qwen-Image-Edit-2511"
save_directory = "/app/model_weights"

print(f"Model indiriliyor: {model_id}")
pipe = QwenImageEditPlusPipeline.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, cache_dir=save_directory
)
pipe.save_pretrained("/app/model_weights")
print("Model başarıyla indirildi ve kaydedildi.")
