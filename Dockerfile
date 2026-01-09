# 1. Aşama: Güncel ve optimize edilmiş bir PyTorch imajı kullan
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 2. Aşama: Çalışma dizinini ayarla
WORKDIR /app

# 3. Aşama: Gerekli kütüphaneleri kur
# Qwen-Image-Edit-2511 gibi modeller için transformers'ın güncel olması kritiktir.
RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    runpod \
    requests \
    pillow

# 4. Aşama: Handler dosyanı ve varsa model ağırlıklarını kopyala
COPY handler.py /app/handler.py

# 5. Aşama: RunPod'un handler'ı tetiklemesini sağla
CMD ["python", "-u", "/app/handler.py"]