import torch
from diffusers import QwenImageEditPlusPipeline
from huggingface_hub import hf_hub_download

# Ana model ID'leri
BASE_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
LIGHTNING_LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LIGHTNING_LORA_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

print("=" * 50)
print("Model indirme başlıyor...")
print("=" * 50)

# 1. Ana modeli indir
print(f"\n[1/2] Ana model indiriliyor: {BASE_MODEL_ID}")
pipe = QwenImageEditPlusPipeline.from_pretrained(
    BASE_MODEL_ID, torch_dtype=torch.bfloat16
)
print(f"✓ Ana model indirildi: {BASE_MODEL_ID}")

# 2. Lightning LoRA'yı indir (4 adımlık hızlı inference için)
print(f"\n[2/2] Lightning LoRA indiriliyor: {LIGHTNING_LORA_REPO}")
lora_path = hf_hub_download(
    repo_id=LIGHTNING_LORA_REPO,
    filename=LIGHTNING_LORA_FILE,
)
print(f"✓ Lightning LoRA indirildi: {lora_path}")

print("\n" + "=" * 50)
print("Tüm modeller başarıyla indirildi!")
print("=" * 50)
