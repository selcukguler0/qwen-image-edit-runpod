import runpod
import torch
import base64
import io
import requests
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# 1. Modeli global alanda yükle (Cold Start sırasında bir kez çalışır)
print("Model yükleniyor... Bu biraz zaman alabilir.")
try:
    # handler.py içinde pipeline yükleme kısmını şu şekilde güncelle:
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "/app/model_weights",  # Cache yapısına göre değişebilir
        torch_dtype=torch.bfloat16,
        local_files_only=True,  # İnternete gitmesini engeller
        device_map="auto",
    )
    # Bellek optimizasyonu için (Opsiyonel)
    # pipe.enable_model_cpu_offload()
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yükleme hatası: {e}")


def decode_image(image_data):
    """Base64 veya URL'den gelen görseli PIL Image'e çevirir."""
    if image_data.startswith("http"):
        response = requests.get(image_data)
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_image(image):
    """PIL Image'i Base64 string'e çevirir."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def handler(event):
    """RunPod'dan gelen her isteği karşılayan ana fonksiyon."""
    try:
        # Input parametrelerini al
        job_input = event.get("input", {})

        prompt = job_input.get("prompt")
        image_1 = job_input.get("image_1")  # URL veya Base64
        image_2 = job_input.get("image_2")  # URL veya Base64 - isteğe bağlı

        # Opsiyonel parametreler (varsayılan değerlerle)
        steps = job_input.get("num_inference_steps", 40)
        guidance_scale = job_input.get("guidance_scale", 1.0)
        seed = job_input.get("seed", 42)

        if not prompt or not image_1:
            return {"error": "Prompt ve image parametreleri zorunludur."}

        # Üretim (Inference)
        generator = torch.manual_seed(seed)

        # Görseli hazırla
        image_1 = decode_image(image_1)

        # 2. Görsel varsa
        if image_2:
            image_2 = decode_image(image_2)
            with torch.inference_mode():
                output = pipe(
                    image=[image_1, image_2],
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    negative_prompt="low quality, blurry, distorted",
                )
        # Tek görsel varsa
        else:
            with torch.inference_mode():
                output = pipe(
                    image=[image_1],
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    negative_prompt="low quality, blurry, distorted",
                )

        final_image = output.images[0]

        # Sonucu Base64 olarak dön
        return {"status": "success", "image": encode_image(final_image)}

    except Exception as e:
        return {"error": str(e)}


# RunPod servisini başlat
runpod.serverless.start({"handler": handler})
