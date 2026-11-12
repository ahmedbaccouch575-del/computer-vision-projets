"""Accès aux données (Google Sheets) + données de démo.

Aucune dépendance à Streamlit : ce module peut être testé seul.
"""
import os
import random
import re
from datetime import date, timedelta

import pandas as pd

SHEET_TITLE = os.getenv("SHEET_TITLE", "DefectGuard AI")
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
RECLAMATION_COLS = [
    "claim_id", "order_id", "date", "photo_url", "description", "defaut_type",
    "gravite", "coherence", "score_fraude", "decision", "statut", "montant",
    "commentaire_humain",
]
AUTO_DECISIONS = ["remplacer", "rembourser", "refuser"]


# --------------------------------------------------------------------------
# Google Sheets
# --------------------------------------------------------------------------
def _client(info=None):
    import gspread
    from google.oauth2.service_account import Credentials

    if info:
        creds = Credentials.from_service_account_info(dict(info), scopes=SCOPES)
    else:
        path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
        creds = Credentials.from_service_account_file(path, scopes=SCOPES)
    return gspread.authorize(creds)


def load_sheets(info=None):
    """Retourne {'commandes','reclamations','clients','stock'} en DataFrames."""
    book = _client(info).open(SHEET_TITLE)
    out = {}
    for name in ["commandes", "reclamations", "clients", "stock"]:
        out[name] = pd.DataFrame(book.worksheet(name).get_all_records())
    return out


def update_claim(claim_id, fields, info=None):
    """Met à jour une ligne de 'reclamations' (recherche par claim_id)."""
    ws = _client(info).open(SHEET_TITLE).worksheet("reclamations")
    headers = ws.row_values(1)
    cell = ws.find(str(claim_id), in_column=headers.index("claim_id") + 1)
    if cell is None:
        raise ValueError(f"claim_id introuvable : {claim_id}")
    for key, value in fields.items():
        if key in headers:
            ws.update_cell(cell.row, headers.index(key) + 1, value)


# --------------------------------------------------------------------------
# Préparation
# --------------------------------------------------------------------------
def drive_image_url(url, width=600):
    """Transforme un lien Drive (uc?id=...) en miniature affichable."""
    m = re.search(r"[?&]id=([\w-]+)", str(url)) or re.search(r"/d/([\w-]+)", str(url))
    return f"https://drive.google.com/thumbnail?id={m.group(1)}&sz=w{width}" if m else None


def prepare(rec, cmd):
    """Joint réclamations + commandes et force les bons types."""
    rec = rec.copy()
    for col in RECLAMATION_COLS:
        if col not in rec.columns:
            rec[col] = ""
    keep = ["order_id", "produit", "categorie", "prix", "fournisseur", "lot", "transporteur"]
    cmd = cmd[[c for c in keep if c in cmd.columns]].drop_duplicates("order_id")
    df = rec.merge(cmd, on="order_id", how="left")
    for col in ["montant", "score_fraude", "coherence", "prix"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["decision", "statut", "defaut_type", "gravite", "produit",
                "fournisseur", "lot", "transporteur"]:
        df[col] = df.get(col, "").fillna("").astype(str)
    return df


def compute_kpis(df):
    total = len(df)
    if total == 0:
        return dict(total=0, a_valider=0, auto_pct=0.0, cout=0.0, fraude_moy=0.0)
    a_valider = int((df["statut"] == "a_valider").sum())
    auto = int(df["statut"].isin(["traitee", "cloture"]).sum())
    return dict(
        total=total,
        a_valider=a_valider,
        auto_pct=round(100 * auto / total, 1),
        cout=float(df["montant"].sum()),
        fraude_moy=round(float(df["score_fraude"].mean()), 1),
    )


# --------------------------------------------------------------------------
# Données de démonstration (utilisées si Google Sheets n'est pas configuré)
# --------------------------------------------------------------------------
def demo_data():
    random.seed(11)
    cmd = pd.DataFrame([
        ("CMD-1001", "Mug céramique blanc", "Maison", 12.9, "CeramPro", "LOT-M-2410", "Aramex"),
        ("CMD-1002", "Casque Bluetooth X200", "Électronique", 59.0, "SoundTech", "LOT-E-2409", "DHL"),
        ("CMD-1003", "Robe d'été fleurie", "Vêtements", 34.5, "ModaTex", "LOT-V-2410", "Aramex"),
        ("CMD-1004", "Smartphone Z5", "Électronique", 249.0, "MobiCorp", "LOT-E-2411", "DHL"),
        ("CMD-1006", "Vase en verre", "Maison", 18.0, "CeramPro", "LOT-M-2410", "Aramex"),
        ("CMD-1007", "Sneakers Urban", "Chaussures", 79.9, "StepWear", "LOT-C-2410", "DHL"),
        ("CMD-1008", "Montre connectée W3", "Électronique", 129.0, "TimeTech", "LOT-E-2411", "DHL"),
        ("CMD-1009", "Set de 6 assiettes", "Maison", 39.0, "CeramPro", "LOT-M-2411", "Aramex"),
        ("CMD-1011", "Écouteurs sans fil Pods", "Électronique", 45.0, "SoundTech", "LOT-E-2409", "DHL"),
        ("CMD-1012", "Sac à dos 25L", "Accessoires", 42.0, "BagLine", "LOT-A-2410", "Aramex"),
    ], columns=["order_id", "produit", "categorie", "prix", "fournisseur", "lot", "transporteur"])

    defauts = {
        "Maison": ["casse", "casse", "raye", "emballage_ouvert"],
        "Électronique": ["casse", "piece_manquante", "aucun", "mauvais_produit"],
        "Vêtements": ["tache", "tache", "mauvais_produit"],
        "Chaussures": ["piece_manquante", "raye"],
        "Accessoires": ["raye", "emballage_ouvert"],
    }
    rows = []
    today = date.today()
    for i in range(30):
        c = cmd.sample(1, random_state=i).iloc[0]
        defaut = random.choice(defauts[c.categorie])
        fraude = random.choice([5, 8, 12, 15, 20, 35, 70, 85]) if defaut != "aucun" else 55
        if fraude > 60 or c.prix > 100:
            decision, statut = "escalader", "a_valider"
        elif defaut == "aucun":
            decision, statut = "refuser", "cloture"
        else:
            decision, statut = random.choice(["remplacer", "rembourser"]), "traitee"
        rows.append(dict(
            claim_id=f"CLM-{26110000 + i}",
            order_id=c.order_id,
            date=(today - timedelta(days=random.randint(0, 20))).isoformat(),
            photo_url="",
            description=f"{c.produit} : problème signalé par le client",
            defaut_type=defaut,
            gravite=random.choice(["faible", "moyenne", "critique"]),
            coherence=random.randint(40, 100),
            score_fraude=fraude,
            decision=decision,
            statut=statut,
            montant=c.prix if decision == "rembourser" else 0,
            commentaire_humain="fraude > 60" if fraude > 60 else ("montant > 100 €" if c.prix > 100 else ""),
        ))
    return {"commandes": cmd, "reclamations": pd.DataFrame(rows),
            "clients": pd.DataFrame(), "stock": pd.DataFrame()}
