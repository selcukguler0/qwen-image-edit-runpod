import base64
from util import post_request

# ============================================================
# Lightning LoRA ile Optimizasyon Testi
# Varsayılan: 4 adım (40 adım yerine = 10x hızlanma)
# ============================================================

# Seçenek 1: URL ile gönder (önerilen - daha hızlı, daha az bant genişliği)
# payload = {
#     "input": {
#         "prompt": "Change the t-shirt to a blue color",
#         "image_1": "https://example.com/your-image.jpg",
#         "num_inference_steps": 4,  # Lightning LoRA ile 4 adım yeterli
#     }
# }

# Seçenek 2: Base64 ile gönder (lokal test için)
payload = {
    "input": {
        "prompt": "Change the t-shirt to a blue color, do not touch background or person face just focus clothe change",
        "image_1": base64.b64encode(open("IMG_8997.JPG", "rb").read()).decode("utf-8"),
        "num_inference_steps": 4,  # Lightning LoRA ile 4 adım yeterli (varsayılan)
        "guidance_scale": 1.0,
    }
}

print("İstek gönderiliyor (Lightning LoRA - 4 steps)...")
response = post_request(payload)
