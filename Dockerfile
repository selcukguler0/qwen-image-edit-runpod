FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Hızlı indirme için HF_TRANSFER'i aktif et
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Gerekli kütüphaneleri kur
# 8-bit (FP8 benzeri) kullanım için 'bitsandbytes' ekledik
RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    runpod \
    requests \
    pillow \
    bitsandbytes \
    hf_transfer

# Önce indirme scriptini kopyala ve MODELLERİ İNDİR
COPY download_model.py .
RUN python -u download_model.py && rm download_model.py

COPY handler.py .
# Artık /app/model_weights gibi ekstra klasörlere gerek kalmadı
CMD ["python", "-u", "handler.py"]