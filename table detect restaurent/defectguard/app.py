"""DefectGuard AI : dashboard SAV (Streamlit)."""
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

import data

load_dotenv()

st.set_page_config(page_title="DefectGuard AI", page_icon="🔍", layout="wide")

WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")


# --------------------------------------------------------------------------
# Chargement des données
# --------------------------------------------------------------------------
def _secrets_info():
    try:
        return dict(st.secrets["gcp_service_account"])
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner="Chargement des données…")
def load():
    try:
        sheets = data.load_sheets(_secrets_info())
        return sheets, False, ""
    except Exception as exc:  # pas de credentials, feuille introuvable, etc.
        return data.demo_data(), True, str(exc)


sheets, is_demo, load_error = load()
df = data.prepare(sheets["reclamations"], sheets["commandes"])

# modifications faites dans cette session (avant rafraîchissement du cache)
overrides = st.session_state.setdefault("overrides", {})
for claim_id, fields in overrides.items():
    for col, value in fields.items():
        df.loc[df["claim_id"] == claim_id, col] = value

# --------------------------------------------------------------------------
# Barre latérale
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 DefectGuard AI")
    if is_demo:
        st.warning("Mode démo : Google Sheets non connecté.")
        with st.expander("Détail"):
            st.code(load_error or "aucune erreur", language="text")
    else:
        st.success("Connecté à Google Sheets")
    if st.button("🔄 Rafraîchir"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    fournisseurs = sorted([f for f in df["fournisseur"].unique() if f])
    sel_f = st.multiselect("Fournisseur", fournisseurs)
    if sel_f:
        df = df[df["fournisseur"].isin(sel_f)]
    if df["date"].notna().any():
        dmin, dmax = df["date"].min().date(), df["date"].max().date()
        if dmin != dmax:
            start, end = st.slider("Période", dmin, dmax, (dmin, dmax))
            df = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]

st.title("Tableau de bord SAV")
tab_file, tab_kpi, tab_pat, tab_chat = st.tabs(
    ["📥 File SAV", "📊 KPIs", "🧩 Patterns", "💬 Chat analytique"]
)

# --------------------------------------------------------------------------
# 1. File SAV (human-in-the-loop)
# --------------------------------------------------------------------------
with tab_file:
    only_pending = st.toggle("Afficher seulement les dossiers à valider", value=True)
    queue = df[df["statut"] == "a_valider"] if only_pending else df
    queue = queue.sort_values("score_fraude", ascending=False)
    st.caption(f"{len(queue)} dossier(s)")

    for _, row in queue.iterrows():
        cid = row["claim_id"]
        title = f"{cid} · {row['produit'] or row['order_id']} · fraude {int(row['score_fraude'])}/100"
        with st.expander(title, expanded=(row["statut"] == "a_valider")):
            left, right = st.columns([1, 2])
            with left:
                img = data.drive_image_url(row["photo_url"])
                if img:
                    st.image(img, width=280)
                else:
                    st.info("Pas de photo")
            with right:
                st.markdown(
                    f"**Commande** {row['order_id']} · **Prix** {row['prix']:.2f} €  \n"
                    f"**Fournisseur** {row['fournisseur']} · **Lot** {row['lot']} · "
                    f"**Transporteur** {row['transporteur']}  \n"
                    f"**Défaut** {row['defaut_type']} ({row['gravite']}) · "
                    f"**Cohérence** {int(row['coherence'])} %  \n"
                    f"**Statut** `{row['statut']}` · **Décision** `{row['decision']}`"
                )
                st.write(f"💬 Client : {row['description']}")
                if row["commentaire_humain"]:
                    st.caption(f"Raison du garde-fou : {row['commentaire_humain']}")

                if row["statut"] == "a_valider":
                    c1, c2 = st.columns([1, 2])
                    final = c1.selectbox(
                        "Décision finale", ["remplacer", "rembourser", "refuser"],
                        key=f"dec_{cid}")
                    comment = c2.text_input("Commentaire", key=f"com_{cid}")
                    if st.button("✅ Confirmer la décision", key=f"ok_{cid}"):
                        montant = float(row["prix"]) if final == "rembourser" else 0
                        statut = "cloture" if final == "refuser" else "traitee"
                        fields = {"decision": final, "statut": statut, "montant": montant,
                                  "commentaire_humain": f"Validé SAV : {comment}"}
                        try:
                            if not is_demo:
                                data.update_claim(cid, fields, _secrets_info())
                            if WEBHOOK_URL:
                                requests.post(WEBHOOK_URL, timeout=15, json={
                                    "claim_id": cid, "order_id": row["order_id"],
                                    "decision": final, "montant": montant,
                                    "comment": comment})
                            overrides[cid] = fields
                            st.success("Décision enregistrée" + (
                                " (démo : rien n'est écrit dans Sheets)" if is_demo else ""))
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Échec : {exc}")

# --------------------------------------------------------------------------
# 2. KPIs
# --------------------------------------------------------------------------
with tab_kpi:
    k = data.compute_kpis(df)
    a, b, c, d, e = st.columns(5)
    a.metric("Réclamations", k["total"])
    b.metric("À valider", k["a_valider"])
    c.metric("Traitées automatiquement", f"{k['auto_pct']} %")
    d.metric("Coût des défauts", f"{k['cout']:.0f} €")
    e.metric("Fraude moyenne", f"{k['fraude_moy']}/100")

    if len(df):
        col1, col2 = st.columns(2)
        per_day = df.groupby(df["date"].dt.date).size().reset_index(name="réclamations")
        col1.plotly_chart(px.bar(per_day, x="date", y="réclamations",
                                 title="Réclamations par jour"))
        per_dec = df.groupby("decision").size().reset_index(name="nombre")
        col2.plotly_chart(px.bar(per_dec, x="decision", y="nombre",
                                 title="Répartition des décisions"))

# --------------------------------------------------------------------------
# 3. Patterns
# --------------------------------------------------------------------------
with tab_pat:
    if df.empty:
        st.info("Aucune donnée.")
    else:
        pivot = df.pivot_table(index="produit", columns="defaut_type",
                               values="claim_id", aggfunc="count", fill_value=0)
        st.plotly_chart(px.imshow(pivot, text_auto=True, aspect="auto",
                                  title="Produit × type de défaut"))
        c1, c2, c3 = st.columns(3)
        for col, field, label in [(c1, "fournisseur", "Fournisseurs"),
                                  (c2, "lot", "Lots"),
                                  (c3, "transporteur", "Transporteurs")]:
            top = (df[df[field] != ""].groupby(field).size()
                   .sort_values(ascending=False).head(8).reset_index(name="défauts"))
            col.plotly_chart(px.bar(top, x="défauts", y=field, orientation="h",
                                    title=f"Top {label.lower()}"))

        lots = df[df["lot"] != ""].groupby(["fournisseur", "lot"]).size().reset_index(name="n")
        worst = lots.sort_values("n", ascending=False).head(1)
        if len(worst) and worst.iloc[0]["n"] >= 3:
            w = worst.iloc[0]
            st.warning(f"⚠️ Lot à surveiller : **{w['lot']}** ({w['fournisseur']}) "
                       f"avec {int(w['n'])} réclamations.")

# --------------------------------------------------------------------------
# 4. Chat analytique (Gemini)
# --------------------------------------------------------------------------
with tab_chat:
    st.caption("Pose une question sur les réclamations en langage naturel.")
    question = st.text_input("Question",
                             placeholder="Quel transporteur casse le plus de colis ?")
    if question:
        if not GEMINI_KEY:
            st.error("Ajoute GEMINI_API_KEY dans le fichier .env pour activer le chat.")
        else:
            cols = ["claim_id", "date", "produit", "categorie", "prix", "fournisseur",
                    "lot", "transporteur", "defaut_type", "gravite", "score_fraude",
                    "decision", "statut", "montant"]
            table = df[cols].assign(date=df["date"].dt.date).head(300).to_csv(index=False)
            prompt = (
                "Tu es analyste SAV. Réponds en français, de façon concise, "
                "uniquement à partir du tableau CSV ci-dessous (réclamations). "
                "Si la réponse n'est pas dans les données, dis-le.\n\n"
                f"{table}\n\nQuestion : {question}")
            try:
                resp = requests.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    "gemini-2.5-flash:generateContent",
                    headers={"x-goog-api-key": GEMINI_KEY},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=60)
                resp.raise_for_status()
                st.write(resp.json()["candidates"][0]["content"]["parts"][0]["text"])
            except Exception as exc:
                st.error(f"Erreur Gemini : {exc}")
