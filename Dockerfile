FROM runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204

WORKDIR /app

# Hızlı indirme için HF_TRANSFER'i aktif et
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Gerekli kütüphaneleri kur
RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    runpod \
    requests \
    pillow \
    hf_transfer

# Model indirme scriptini kopyala ve modelleri indir
COPY download_model.py .
RUN python -u download_model.py && rm download_model.py

# Handler'ı kopyala
COPY handler.py .

# Servisi başlat
CMD ["python", "-u", "handler.py"]