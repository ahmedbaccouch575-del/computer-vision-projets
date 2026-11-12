FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Librairies système nécessaires à OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch version CPU (beaucoup plus léger que la version GPU)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Code et modèles (la vidéo n'est PAS copiée : elle est montée au lancement)
COPY phase/ phase/
COPY utils/ utils/
COPY best.pt yolo11s.pt ./

CMD ["python", "phase/model.py"]