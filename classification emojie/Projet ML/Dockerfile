FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application et le modèle déjà entraîné
COPY app.py .
COPY model/ ./model/

EXPOSE 5000

# Utiliser gunicorn en production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]