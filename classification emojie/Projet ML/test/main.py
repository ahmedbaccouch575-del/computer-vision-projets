"""
test/main.py
API Flask pour servir le modèle.

Utilisation (depuis la racine du projet) :
    python test/main.py

Endpoints :
    GET  /            → message d'accueil
    GET  /columns     → liste des colonnes attendues
    POST /predict     → prédiction
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Remonter d'un dossier pour trouver ../model/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

# Chargement des artefacts
model   = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API de prédiction du cancer du sein",
        "endpoints": {
            "/predict": "POST - fait une prédiction",
            "/columns": "GET - liste les colonnes attendues"
        }
    })


@app.route("/columns", methods=["GET"])
def get_columns():
    return jsonify({"columns": columns, "count": len(columns)})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if not payload or "features" not in payload:
        return jsonify({"error": "Le champ 'features' est requis."}), 400

    features = payload["features"]

    try:
        if isinstance(features, dict):
            missing = [c for c in columns if c not in features]
            if missing:
                return jsonify({"error": f"Colonnes manquantes : {missing}"}), 400
            row = [features[c] for c in columns]
            X = pd.DataFrame([row], columns=columns)

        elif isinstance(features, list):
            if len(features) != len(columns):
                return jsonify({
                    "error": f"Attendu {len(columns)} features, reçu {len(features)}"
                }), 400
            X = pd.DataFrame([features], columns=columns)

        else:
            return jsonify({"error": "'features' doit être objet ou liste."}), 400

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Données invalides : {e}"}), 400

    # Prédiction
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    diagnosis = "M" if prediction == 1 else "B"
    return jsonify({
        "diagnosis": diagnosis,
        "diagnosis_label": "Malin" if diagnosis == "M" else "Bénin",
        "prediction": int(prediction),
        "probability": {
            "B": round(float(proba[0]), 4),
            "M": round(float(proba[1]), 4)
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)