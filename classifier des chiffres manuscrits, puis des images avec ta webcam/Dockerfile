FROM python:3.11-slim

WORKDIR /app

# Installer d'abord les dependances (meilleur cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier ensuite le code
COPY main.py .

CMD ["python", "main.py"]