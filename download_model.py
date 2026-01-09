import torch
import os
from diffusers import QwenImageEditPlusPipeline

# Modeli belirtilen dizine indir
model_id = "Qwen/Qwen-Image-Edit-2511"
save_directory = "/app/model_weights"

print(f"Model indiriliyor: {model_id}...")

# Modeli indir
pipe = QwenImageEditPlusPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    # cache_dir kullanmak yerine doğrudan indirmeyi yönetiyoruz
)

# Dosyaları snapshot karmaşasından kurtarıp düzgünce kaydet
pipe.save_pretrained(save_directory)
print(f"Model başarıyla {save_directory} dizinine kaydedildi.")
