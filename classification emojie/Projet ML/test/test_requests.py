"""
test/test_requests.py
Script de test pour l'API Flask de prédiction du cancer du sein.

Prérequis : lancer l'API avant d'exécuter ce script
    python test/main.py

Utilisation :
    python test/test_requests.py
"""

import requests
import sys

BASE_URL = "http://127.0.0.1:5000"

PASSED = []
FAILED = []


def check(name, condition, detail=""):
    """Affiche clairement si un test réussit ou échoue."""
    if condition:
        print(f"[OK]   {name}")
        PASSED.append(name)
    else:
        print(f"[FAIL] {name} {detail}")
        FAILED.append(name)

# Exemple de features (valeurs réelles issues du dataset, patient M)
SAMPLE_FEATURES = {
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.6,
    "area_worst": 2019,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189,
}


def test_home():
    print("=== GET / ===")
    r = requests.get(f"{BASE_URL}/")
    print(r.status_code, r.json())
    check("GET / renvoie 200", r.status_code == 200, f"(reçu {r.status_code})")
    print()


def test_columns():
    print("=== GET /columns ===")
    r = requests.get(f"{BASE_URL}/columns")
    data = r.json()
    print(r.status_code, data)
    cols = data.get("columns", [])
    check("GET /columns renvoie 200", r.status_code == 200, f"(reçu {r.status_code})")
    check("GET /columns renvoie 30 colonnes", len(cols) == 30, f"(reçu {len(cols)})")
    print()
    return cols


def test_predict_with_dict():
    # Cas normal : on ENVOIE des données valides -> on ATTEND un succès (200)
    print("=== POST /predict (dict) — cas valide, doit réussir ===")
    r = requests.post(f"{BASE_URL}/predict", json={"features": SAMPLE_FEATURES})
    data = r.json()
    print(r.status_code, data)
    check("predict(dict) renvoie 200", r.status_code == 200, f"(reçu {r.status_code})")
    check("predict(dict) diagnostic = M", data.get("diagnosis") == "M", f"(reçu {data.get('diagnosis')})")
    print()


def test_predict_with_list(columns):
    # Cas normal : on ENVOIE des données valides -> on ATTEND un succès (200)
    print("=== POST /predict (list) — cas valide, doit réussir ===")
    if not columns:
        print("Colonnes non récupérées, test ignoré.")
        FAILED.append("predict(list) - colonnes indisponibles")
        return
    values = [SAMPLE_FEATURES[c] for c in columns]
    r = requests.post(f"{BASE_URL}/predict", json={"features": values})
    data = r.json()
    print(r.status_code, data)
    check("predict(list) renvoie 200", r.status_code == 200, f"(reçu {r.status_code})")
    check("predict(list) diagnostic = M", data.get("diagnosis") == "M", f"(reçu {data.get('diagnosis')})")
    print()


def test_predict_missing_field():
    # Cas volontairement invalide : on ENVOIE un JSON vide -> on ATTEND un rejet (400)
    print("=== POST /predict (champ 'features' manquant) — doit être REJETÉ (400) ===")
    r = requests.post(f"{BASE_URL}/predict", json={})
    print(r.status_code, r.json())
    check("champ manquant -> 400 (rejet attendu)", r.status_code == 400, f"(reçu {r.status_code})")
    print()


def test_predict_wrong_length():
    # Cas volontairement invalide : on ENVOIE trop peu de valeurs -> on ATTEND un rejet (400)
    print("=== POST /predict (liste trop courte) — doit être REJETÉ (400) ===")
    r = requests.post(f"{BASE_URL}/predict", json={"features": [1, 2, 3]})
    print(r.status_code, r.json())
    check("liste trop courte -> 400 (rejet attendu)", r.status_code == 400, f"(reçu {r.status_code})")
    print()


if __name__ == "__main__":
    test_home()
    cols = test_columns()
    test_predict_with_dict()
    test_predict_with_list(cols)
    test_predict_missing_field()
    test_predict_wrong_length()

    print("=" * 50)
    print(f"Résultat : {len(PASSED)} réussis, {len(FAILED)} échoués")
    if FAILED:
        print("Échecs :", ", ".join(FAILED))
        sys.exit(1)
    else:
        print("Tous les tests sont OK. Rien n'est cassé.")
        sys.exit(0)