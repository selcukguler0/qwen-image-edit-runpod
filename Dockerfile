FROM runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204

WORKDIR /app

# Hızlı indirme için HF_TRANSFER'i aktif et
ENV HF_HUB_ENABLE_HF_TRANSFER=1

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