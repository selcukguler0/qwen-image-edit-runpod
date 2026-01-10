import base64
from util import post_request

payload = {
    "input": {
        "prompt": "Change the t-shirt to a blue color, do not touch background or person face just focus clothe change",
        "image_1": base64.b64encode(open("IMG_8997.JPG", "rb").read()).decode("utf-8"),
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
    }
}


print("İstek gönderiliyor...")
response = post_request(payload)
