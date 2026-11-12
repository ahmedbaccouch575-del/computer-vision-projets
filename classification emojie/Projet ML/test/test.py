

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Remonter d'un dossier (on est dans test/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "breast-cancer.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Charger le CSV
df = pd.read_csv(CSV_PATH)

# 2. Séparer features / cible
X = df.drop(columns=["id", "diagnosis"])
y = df["diagnosis"].map({"B": 0, "M": 1})   # 0 = Bénin, 1 = Malin

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Modèle
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

print(f"Accuracy sur le test : {model.score(X_test_scaled, y_test):.4f}")

# 6. Sauvegarde
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "columns.pkl"))

print(f"✅ Modèle sauvegardé dans {MODEL_DIR}")