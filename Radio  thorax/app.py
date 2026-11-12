from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms as T
from PIL import Image

# Chemin calculé à partir de l'emplacement d'app.py (marche quel que soit le dossier de lancement)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"

THRESHOLD = 0.5            # remplace par ton THRESHOLD de l'étape 15
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

st.set_page_config(page_title="Radio thorax", page_icon="🫁", layout="centered")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


@st.cache_resource
def load_model():
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 1)
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    return m


def predict(model, image: Image.Image) -> float:
    x = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        return torch.sigmoid(model(x)).item()


st.title("🫁 Détection de pneumonie sur radio thorax")
st.warning("Outil d'aide au tri uniquement. Il ne remplace pas un avis médical.")

# Vérification claire si le modèle est introuvable
if not MODEL_PATH.exists():
    st.error(f"Modèle introuvable : {MODEL_PATH}")
    st.info(
        "Copie le fichier **best_model.pth** (téléchargé depuis Google Drive) "
        "dans le dossier **models** à côté de app.py, puis relance l'application."
    )
    existing = [p.name for p in BASE_DIR.iterdir()]
    st.caption(f"Contenu du dossier de l'application : {existing}")
    st.stop()

file = st.file_uploader("Charge une radio (jpg, jpeg ou png)", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file)
    st.image(image, caption="Radio chargée", use_container_width=True)

    if st.button("Analyser", type="primary"):
        model = load_model()
        with st.spinner("Analyse en cours..."):
            prob = predict(model, image)

        st.metric("Probabilité de pneumonie", f"{prob * 100:.1f} %")
        st.progress(min(max(prob, 0.0), 1.0))

        if prob >= THRESHOLD:
            st.error("Résultat : pneumonie suspectée. Un avis médical est nécessaire.")
        else:
            st.success("Résultat : pas de signe de pneumonie détecté.")

        st.caption(f"Seuil de décision : {THRESHOLD}")

st.divider()
st.caption(
    "Limites : modèle entraîné sur un seul jeu de données (Kaggle, Kermany), "
    "sans validation clinique. Ne pas utiliser pour un diagnostic."
)