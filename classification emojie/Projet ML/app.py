

import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Chargement du modèle, du scaler et des colonnes au démarrage
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
columns = joblib.load("model/columns.pkl")


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
        return jsonify({"error": "Le champ 'features' est requis dans le JSON."}), 400

    features = payload["features"]

    try:
        if isinstance(features, dict):
            # Reconstruire un DataFrame en respectant l'ordre des colonnes
            missing = [c for c in columns if c not in features]
            if missing:
                return jsonify({"error": f"Colonnes manquantes : {missing}"}), 400
            row = [features[c] for c in columns]
            X = pd.DataFrame([row], columns=columns)

        elif isinstance(features, list):
            if len(features) != len(columns):
                return jsonify({
                    "error": f"Nombre de features incorrect : attendu {len(columns)}, reçu {len(features)}"
                }), 400
            X = pd.DataFrame([features], columns=columns)

        else:
            return jsonify({"error": "'features' doit être un objet ou une liste."}), 400

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Données invalides : {e}"}), 400

    # Standardisation + prédiction
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    diagnosis = "M" if prediction == 1 else "B"
    result = {
        "diagnosis": diagnosis,
        "diagnosis_label": "Malin" if diagnosis == "M" else "Bénin",
        "prediction": int(prediction),
        "probability": {
            "B": round(float(proba[0]), 4),
            "M": round(float(proba[1]), 4)
        }
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)