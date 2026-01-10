import runpod
import torch
import base64
import io
import os
import requests
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from huggingface_hub import hf_hub_download

# CUDA bellek fragmentasyonunu önlemek için
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Maksimum görsel boyutu (piksel) - büyük görseller yeniden boyutlandırılır
MAX_IMAGE_SIZE = 1024

# Lightning LoRA dosya adı
LIGHTNING_LORA_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
LIGHTNING_LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"

print("Model yükleniyor... Bu biraz zaman alabilir.")

try:
    # Ana pipeline'ı yükle
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    # Lightning LoRA'yı yükle - 4 adımda yüksek kaliteli sonuç
    lora_path = hf_hub_download(
        repo_id=LIGHTNING_LORA_REPO,
        filename=LIGHTNING_LORA_FILE,
        local_files_only=True,
    )
    pipe.load_lora_weights(lora_path)
    print(f"Lightning LoRA yüklendi: {LIGHTNING_LORA_FILE}")

    # Modeli GPU'ya taşı (A100 için en hızlı yöntem)
    pipe.to("cuda")

    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yükleme hatası: {e}")
    raise e


def resize_image(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """Görseli en-boy oranını koruyarak maksimum boyuta göre yeniden boyutlandırır."""
    width, height = image.size

    # Eğer görsel zaten yeterince küçükse, dokunma
    if width <= max_size and height <= max_size:
        return image

    # En-boy oranını koru
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    # Yüksek kaliteli yeniden boyutlandırma
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(f"Görsel boyutlandırıldı: {width}x{height} -> {new_width}x{new_height}")
    return resized


def process_input(image_data: str) -> Image.Image:
    """URL veya Base64'ten gelen görseli işler ve optimize eder."""
    # URL kontrolü
    if image_data.startswith("http://") or image_data.startswith("https://"):
        response = requests.get(image_data, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        # Base64 decode
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Görseli yeniden boyutlandır
    return resize_image(image)


def encode_image(image: Image.Image) -> str:
    """PIL Image'i Base64 string'e çevirir."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def handler(event):
    """RunPod'dan gelen her isteği karşılayan ana fonksiyon."""
    try:
        job_input = event.get("input", {})

        prompt = job_input.get("prompt")
        image_1 = job_input.get("image_1")
        image_2 = job_input.get("image_2")

        # Lightning LoRA ile varsayılan 4 adım (10x hızlanma)
        steps = job_input.get("num_inference_steps", 4)
        guidance_scale = job_input.get("guidance_scale", 1.0)
        seed = job_input.get("seed", 42)

        if not prompt or not image_1:
            return {"error": "Prompt ve image_1 parametreleri zorunludur."}

        # Görselleri işle ve optimize et
        image_1_processed = process_input(image_1)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Inference
        with torch.inference_mode():
            if image_2:
                image_2_processed = process_input(image_2)
                output = pipe(
                    image=[image_1_processed, image_2_processed],
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                )
            else:
                output = pipe(
                    image=[image_1_processed],
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                )

        final_image = output.images[0]

        return {"status": "success", "image": encode_image(final_image)}

    except Exception as e:
        return {"error": str(e)}


# RunPod servisini başlat
runpod.serverless.start({"handler": handler})
