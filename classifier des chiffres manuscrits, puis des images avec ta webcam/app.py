"""Application Streamlit : reconnaissance de chiffres manuscrits.

Prerequis : avoir lance main.py pour obtenir model.pth
Lancer    : streamlit run app.py
"""
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from main import CNN  # meme architecture que pour l'entrainement

MEAN, STD = 0.1307, 0.3081  # meme normalisation que pour l'entrainement

st.set_page_config(page_title="Reconnaissance de chiffres", page_icon="🔢")


@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model


def preprocess(img):
    """Transforme une photo en image 28x28, chiffre blanc sur fond noir (comme MNIST)."""
    gray = img.convert("L")

    # 1. Traits plus sombres que leur voisinage = le chiffre (marche meme avec un crayon fin
    #    et une lumiere inegale). Le flou sert a estimer la couleur du papier.
    radius = max(5, min(gray.size) // 8)
    background = np.array(gray.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    arr = np.clip(background - np.array(gray, dtype=np.float32), 0, None)
    if arr.max() < 10:  # rien de plus sombre que le fond
        return None
    arr = arr / arr.max() * 255
    arr[arr < 0.25 * 255] = 0  # enlever le bruit

    # 2. Decouper autour du chiffre
    ys, xs = np.where(arr > 0)
    if len(ys) == 0:
        return None
    digit = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 3. Mettre le chiffre dans un carre, centre
    h, w = digit.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top + h, left:left + w] = digit

    # 4. Comme MNIST : chiffre de 20x20 au centre d'une image 28x28
    small = Image.fromarray(square.astype(np.uint8)).resize((20, 20), Image.LANCZOS)
    small = np.array(small, dtype=np.float32)
    small = small / max(small.max(), 1) * 255  # retablir le contraste apres reduction
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(Image.fromarray(small.astype(np.uint8)), (4, 4))

    # 5. Epaissir legerement le trait (les chiffres MNIST ont un trait epais)
    return canvas.filter(ImageFilter.MaxFilter(3))


def predict(model, img28):
    x = torch.from_numpy(np.array(img28, dtype=np.float32) / 255.0)
    x = ((x - MEAN) / STD).unsqueeze(0).unsqueeze(0)  # forme [1, 1, 28, 28]
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    return probs.numpy()


def show_result(model, img, key):
    st.caption("Recadre la zone pour garder seulement le chiffre (enleve visage, fond, bords).")
    x0, x1 = st.slider("Gauche / droite (%)", 0, 100, (0, 100), key=f"x_{key}")
    y0, y1 = st.slider("Haut / bas (%)", 0, 100, (0, 100), key=f"y_{key}")
    if x1 - x0 < 5 or y1 - y0 < 5:
        st.warning("La zone est trop petite.")
        return

    w, h = img.size
    zone = img.crop((int(w * x0 / 100), int(h * y0 / 100), int(w * x1 / 100), int(h * y1 / 100)))
    st.image(zone, caption="Zone analysee", width=250)

    img28 = preprocess(zone)
    if img28 is None:
        st.warning("Aucun chiffre detecte. Essaie avec un trait plus epais et plus contraste.")
        return
    probs = predict(model, img28)
    digit = int(probs.argmax())

    col1, col2 = st.columns(2)
    col1.image(img28.resize((140, 140), Image.NEAREST), caption="Ce que voit le modele (28x28)")
    col2.metric("Chiffre predit", digit, f"{probs[digit] * 100:.1f} % de confiance")
    st.bar_chart({str(i): float(p) for i, p in enumerate(probs)})


st.title("🔢 Reconnaissance de chiffres manuscrits")
st.write("Ecris un chiffre (0 a 9) **en gros, avec un feutre epais**, sur une feuille blanche, "
         "puis envoie une photo ou utilise la camera.")

try:
    model = load_model()
except FileNotFoundError:
    st.error("model.pth introuvable. Lance d'abord `python main.py` pour entrainer le modele.")
    st.stop()

tab_upload, tab_camera = st.tabs(["📁 Importer une image", "📷 Camera"])

with tab_upload:
    file = st.file_uploader("Choisis une image", type=["png", "jpg", "jpeg"])
    if file is not None:
        show_result(model, Image.open(file), key="upload")

with tab_camera:
    photo = st.camera_input("Prends une photo du chiffre")
    if photo is not None:
        show_result(model, Image.open(photo), key="camera")