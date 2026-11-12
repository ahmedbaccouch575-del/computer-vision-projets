"""
Tennis Analyzer - Etape 2 (corrigee + rapide CPU) : joueurs + vitesses + BALLE (test sur video)

Corrections :
  1. Les vitesses n'apparaissent plus sur le public : l'analyse se met en PAUSE
     quand la camera n'est plus sur la vue du court (gros plan, public, replay),
     car la calibration des 4 coins n'est valable que sur la vue large.
  2. Filtre joueur plus strict (marges reduites, on garde le plus grand cadre).
  3. Plus rapide : GPU automatique, STRIDE (1 image sur N), modele/resolution reglables,
     affichage optionnel, progression en console.
  4. Texte des vitesses proportionnel a la taille de la video.

Usage :
    pip install ultralytics opencv-python numpy
    python tennis_analyzer.py ..\\match.mp4

Clique les 4 coins du court sur la 1ere image (VUE LARGE du court) :
    1) loin-gauche   2) loin-droit   3) proche-droit   4) proche-gauche
Appuie sur 'q' pendant la lecture pour arreter.
Sortie : output_tennis.mp4 + resume des vitesses max dans la console.
"""
import sys
import time
import logging
from collections import deque

import os
import unicodedata

import cv2
import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont     # texte net et joli (pip install pillow)
except ImportError:
    Image = None
from ultralytics import YOLO

# ---------------------------------------------------------------
# REGLAGES
# ---------------------------------------------------------------
COURT_W, COURT_L = 10.97, 23.77   # dimensions reelles du court de double (metres)
MAX_HUMAN_SPEED = 8.5             # m/s (~30 km/h) : au-dela, erreur de detection (joueur) ; record d'un sprint de tennis ~ 8 m/s
MAX_HUMAN_SPEED_OLD = 10.0            # m/s : au-dela, erreur de detection (joueur)
WINDOW = 5                        # images traitees pour calculer la vitesse d'un joueur
MINIMAP_HEIGHT = 0.40             # hauteur de la mini-carte = 40 % de la hauteur de la video
MINIMAP_FLIP = False              # True = carte retournee (le joueur proche en haut)
MINIMAP_Y = 0.28                  # position verticale (0 = haut, 1 = bas) ; elle est a droite
# --- Etape 4 : VAR (rebond de la balle -> IN / OUT) ---
VAR_ENABLED = True
VAR_WORD_IN = "IN"                # mot affiche quand la balle est dedans : "IN" ou "GOAL" (si tu preferes)
VAR_CSV = "var_tennis.csv"        # un rebond par ligne
VAR_DIR = "var_images"            # une image zoomee par decision
VAR_SHOW_SECONDS = 3.0            # duree d'affichage de la decision dans la video
VAR_AUTO_PAUSE = False            # True = la fenetre se met en pause a chaque balle OUT (tu peux reculer et zoomer)
BOUNCE_KICK = 0.0045              # saut de vitesse verticale (fraction de la largeur de l'image par image) = rebond
BOUNCE_MIN_SPEED = 0.004          # vitesse min de la balle (fraction de la largeur par image) pour juger
BOUNCE_COS = 0.25                 # la balle garde sa direction (un coup de raquette la retourne : pas un rebond)
BOUNCE_MAX_GAP = 2                # images max sans voir la balle entre 2 points utilises
VAR_NEEDS_RALLY = True            # ne juge un rebond que pendant un echange detecte (evite les decisions sur de fausses balles)
BOUNCE_MIN_RUN = 5                # nombre min d'images de suite ou la balle doit etre vue avant de juger
PLAYER_HEIGHT_RANGE = (0.78, 1.60)  # hauteur de la boite / hauteur attendue d'un adulte (1.8 m) a cet endroit : hors de cet intervalle = ramasseur de balles, enfant, faux
START_X, START_Y_FAR, START_Y_NEAR = 2.0, 4.0, 7.0   # zone ou un joueur peut etre CHOISI (m) ; large cote camera : le joueur proche court loin derriere sa ligne
COAST_SECONDS = 1.0               # si le joueur n'est plus detecte, on garde son etiquette (derniere position) pendant N secondes
PERSON_CONF_TRACK = 0.15          # confiance minimale pour SUIVRE un joueur deja verrouille (penche, de dos, flou)
SPEED_POS_EMA = 0.55              # lissage de la position (0 = tres lisse, 1 = brut)
SPEED_EMA = 0.45                  # lissage de la vitesse affichee
START_Y_UNUSED = 2.5       # un joueur ne peut etre CHOISI que dans cette zone autour des lignes (metres) ; une fois suivi, la zone large s'applique
PLAYER_REACH = 0.55               # zone autour d'un joueur (x sa hauteur) ou une tache jaune = raquette / chaussure, pas la balle
VAR_NEAR_EXTRA = 0.35             # m : incertitude en plus pour les rebonds tout pres de la camera (la balle y fait de tres grands pas entre 2 images)
TAU_TOL = 0.9                     # tolerance (en images) sur l'instant du rebond calcule
VAR_IMPACT_MODE = "4pt"           # "4pt" (4 points, le plus fiable en test), "fit" (ajustement sur 6 points) ou "mean" (moyenne des deux)
BALL_X_MARGIN = 3.5               # m : une balle plus loin que ca sur le cote du court = gradins / decor, pas la balle
BALL_Y_MAX_MARGIN = 6.0           # m : au-dela de la ligne de fond du camp proche
VAR_MAX_OUT = 2.5                 # m : un rebond a plus de 2.5 m des lignes est refuse (fausse balle, pas une decision)
VAR_MODEL_ERR = 0.15               # m : erreur du modele (images espacees, gravite approchee) ajoutee a l'erreur des pixels
SIGMA_PX = 2.5                    # incertitude de position de la balle (pixels) : sert a calculer le "+/- cm" de chaque decision
BALL_RADIUS = 0.033               # m : la balle est IN si elle touche la ligne (rayon de la balle)
LIMIT_NEAR, LIMIT_FAR = 0.15, 0.30   # m : en dessous de cette distance a la ligne, la decision est "limite"
COLOR_BALL = True                 # detecte aussi la balle par sa couleur jaune-vert + mouvement

# --- Vitesse d'execution ---
MODEL_NAME = "yolov8s.pt"         # CPU : "yolov8n.pt" (tres rapide) ou "yolov8s.pt" (compromis) ; GPU : "yolov8m.pt"
IMG_SIZE = 640                    # CPU : 640 ; GPU : 960 ou 1280 (balle mieux vue)
STRIDE = 3                        # CPU : 3 (une image sur trois) ; GPU : 1 ou 2
CLICK_SIGMA_PX = 2.0                # precision du clic manuel VAR (pixels)
SHOW_COURT_OUTLINE = True          # trace le terrain de detection (les 2 joueurs seulement sont analyses)
SHOW_WINDOW = True                # True = fenetre en direct (touche q pour quitter) ; False = plus rapide
BUFFER_FRAMES = 300               # nombre d'images gardees en memoire pour pouvoir reculer (300 images traitees ~ 30 s avec STRIDE=3)
CROP_TO_COURT = True              # detecte seulement autour du court : plus rapide ET joueurs lointains mieux vus

# Coins du court (pixels de la video). None = tu les cliques a la souris.
# Apres un 1er lancement, la console affiche la ligne CORNERS = [...] : colle-la ici
# pour ne plus avoir a cliquer (indispensable sur Colab, qui n'a pas de fenetre).
CORNERS = None
# Calibration PRECISE (conseillee) : 6 points sur les lignes, donnes par la souris au 1er lancement.
# La console affiche alors une ligne CALIB_POINTS = [...] : colle-la ici. (Mets CORNERS = None.)
CALIB_POINTS = None
CALIB_MODE = "4"                  # "4" = clique les 4 coins (comme avant) ; "6" = calibration guidee a 6 points (optionnelle)

PERSON_CONF = 0.30                # confiance minimale pour un joueur
BALL_CONF = 0.10                  # confiance minimale d'un candidat balle
BALL_START_CONF = 0.25            # confiance minimale pour DEMARRER le suivi (evite les fausses balles : chaussures, raquettes)
STATIC_MAX = 8                    # un 'candidat' qui reste au meme endroit plus de N images = objet fixe (balles au sol, seau...) : ignore

# --- Joueurs : zone acceptee autour du court (metres) ---
PLAYER_MARGIN_X = 1.5             # marge a gauche/droite du court
PLAYER_MARGIN_Y_FAR, PLAYER_MARGIN_Y_NEAR = 5.0, 8.0   # marge derriere les lignes de fond (m)
LOST_SECONDS = 2.0                # un joueur non vu pendant plus de N secondes est 'perdu' (on le re-cherche)

# --- Detection de la vue du court (pause de l'analyse hors vue large) ---
MAIN_VIEW_MIN = 0.35              # part min du court visible avec la bonne couleur (0 a 1)
                                  # si l'analyse se coupe trop : baisse (0.25) ; si elle reste sur le public : monte (0.5)

# --- Etape 3 : possession + rapidite du jeu ---
RALLY_TIMEOUT = 2.5               # secondes sans voir la balle = fin de l'echange
HIT_COS = -0.2                    # une frappe = la balle fait demi-tour (angle > ~100 degres)
HIT_MIN_MOVE = 0.012              # deplacement min de la balle entre 2 detections (fraction de la largeur de l'image)
HIT_DEBOUNCE = 0.35               # secondes minimum entre deux frappes
HIT_SAME_PLAYER = 1.5             # secondes : 2 frappes du MEME joueur de suite = doublon, ignore
STATS_CSV = "stats_tennis.csv"    # fichier de resultats (un echange par ligne)

BALL_CLASS = 32                   # classe "sports ball" dans YOLO (COCO)
BALL_JUMP_FRAC = 0.08             # saut max de la balle par image video (fraction de la largeur), multiplie par STRIDE puis plafonne
BALL_MAX_MISSED = 2               # images max ou on "devine" la balle quand elle disparait
BALL_WINDOW = 3                   # images pour calculer la vitesse de la balle
BALL_MAX_SPEED = 65.0             # m/s (~234 km/h) : au-dela, valeur ignoree
TRAIL_LEN = 12                    # longueur de la traînee de la balle

BALL_MODEL = None                 # chemin d'un modele entraine sur la balle (ex: "runs/detect/train/weights/best.pt") ; None = YOLO standard
BALL_MODEL_CLASS = 0              # numero de la classe "balle" dans ce modele
BALL_IMG_SIZE = 1280              # resolution de la passe "balle" (plus grand = balle mieux vue)


def _env_override():
    """Colab : le notebook regle ces valeurs via des variables TV_xxx (aucun fichier a modifier)."""
    import json
    g = globals()
    for k in ("MODEL_NAME", "IMG_SIZE", "STRIDE", "SHOW_WINDOW", "WINDOW", "COLOR_BALL", "CORNERS", "CALIB_POINTS", "CALIB_MODE", "BALL_CONF",
              "BALL_START_CONF", "PERSON_CONF", "VAR_AUTO_PAUSE", "MAIN_VIEW_MIN", "MAX_HUMAN_SPEED", "VAR_WORD_IN",
              "CROP_TO_COURT", "BALL_MODEL", "BALL_MODEL_CLASS", "BALL_IMG_SIZE", "BUFFER_FRAMES"):
        v = os.environ.get("TV_" + k)
        if v is not None:
            g[k] = v if k in ("MODEL_NAME", "VAR_WORD_IN", "BALL_MODEL") else json.loads(v)


_env_override()

logging.getLogger("ultralytics").setLevel(logging.ERROR)   # moins de messages

# GPU automatique
try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"


# ---------------------------------------------------------------
# FONCTION 1 : choisir les 4 coins du court avec la souris
# ---------------------------------------------------------------
def pick_corners(frame):
    w = frame.shape[1]
    s = min(1.0, 1280 / w)        # reduction pour que l'image tienne sur l'ecran
    pts = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    win = "Clique les 4 coins du court (ESC pour annuler)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    labels = ["loin-gauche", "loin-droit", "proche-droit", "proche-gauche"]

    while True:
        img = cv2.resize(frame, None, fx=s, fy=s)
        for i, p in enumerate(pts):
            cv2.circle(img, p, 6, (0, 0, 255), -1)
            cv2.putText(img, labels[i], (p[0] + 8, p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        if len(pts) < 4:
            cv2.putText(img, f"Coin : {labels[len(pts)]}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if len(pts) == 4:
            break
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit("Annule.")

    cv2.destroyWindow(win)
    res = [(round(x / s), round(y / s)) for x, y in pts]   # remet a l'echelle d'origine
    print("\nCoins choisis -> colle cette ligne dans le fichier pour ne plus cliquer :")
    print(f"CORNERS = {res}\n")
    return np.float32(res)


# ---------------------------------------------------------------
# FONCTION 1a : calibration PRECISE par 6 points sur les lignes (toujours visibles en retransmission)
#   1 coin loin-gauche  2 coin loin-droit  (lignes de fond, couloirs de double)
#   3-4 ligne de service LOINTAINE x ligne de simple (gauche, droite)
#   5-6 ligne de service PROCHE    x ligne de simple (gauche, droite)
# Plus fiable que les 4 coins (le fond du court proche est souvent hors image) + loupe + apercu a valider.
# ---------------------------------------------------------------
SX0, SX1 = 1.37, COURT_W - 1.37                     # lignes de simple
SY_FAR, SY_NEAR = COURT_L / 2 - 6.40, COURT_L / 2 + 6.40   # lignes de service
CALIB_WORLD = [(0, 0), (COURT_W, 0), (SX0, SY_FAR), (SX1, SY_FAR), (SX0, SY_NEAR), (SX1, SY_NEAR)]
CALIB_LABELS = ["1/6  COIN LOIN-GAUCHE (ligne de fond x couloir de double)",
                "2/6  COIN LOIN-DROIT (ligne de fond x couloir de double)",
                "3/6  ligne de service LOINTAINE x ligne de simple GAUCHE",
                "4/6  ligne de service LOINTAINE x ligne de simple DROITE",
                "5/6  ligne de service PROCHE x ligne de simple GAUCHE",
                "6/6  ligne de service PROCHE x ligne de simple DROITE"]


def calib_from_points(pts):
    """6 points image -> homographie (image -> metres) ajustee par moindres carres, coins du court dans l'image, erreur (px)."""
    src = np.float32(pts)
    dst = np.float32(CALIB_WORLD)
    H, _ = cv2.findHomography(src, dst, 0)
    Hi = np.linalg.inv(H)
    back = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), Hi).reshape(-1, 2)
    err = float(np.mean(np.hypot(*(back - src).T)))
    full = np.float32([(0, 0), (COURT_W, 0), (COURT_W, COURT_L), (0, COURT_L)])
    corners = cv2.perspectiveTransform(full.reshape(-1, 1, 2), Hi).reshape(-1, 2)
    return H, np.float32(corners), err


def draw_court_lines(img, H, color=(255, 220, 0), th=1):
    """Dessine toutes les lignes du court (calculees depuis la calibration) pour verifier l'alignement."""
    Hi = np.linalg.inv(H)
    segs = [((0, 0), (COURT_W, 0)), ((COURT_W, 0), (COURT_W, COURT_L)), ((COURT_W, COURT_L), (0, COURT_L)), ((0, COURT_L), (0, 0)),
            ((SX0, 0), (SX0, COURT_L)), ((SX1, 0), (SX1, COURT_L)),
            ((SX0, SY_FAR), (SX1, SY_FAR)), ((SX0, SY_NEAR), (SX1, SY_NEAR)),
            ((COURT_W / 2, SY_FAR), (COURT_W / 2, SY_NEAR)), ((0, COURT_L / 2), (COURT_W, COURT_L / 2))]
    for a, b in segs:
        pa = cv2.perspectiveTransform(np.float32([[a]]), Hi)[0, 0]
        pb = cv2.perspectiveTransform(np.float32([[b]]), Hi)[0, 0]
        if max(abs(pa).max(), abs(pb).max()) < 1e5:
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, th, cv2.LINE_AA)


def _click_ok(pts, p, fh):
    """Verifie qu'un nouveau clic est coherent avec les precedents. Retourne un message d'erreur ou None."""
    i = len(pts)
    if i == 1 and not (p[0] > pts[0][0] + 20 and abs(p[1] - pts[0][1]) < 0.12 * fh):
        return "Le point 2 doit etre a DROITE du point 1, sur la meme ligne de fond (loin)"
    if i == 2 and p[1] <= pts[0][1] + 5:
        return "Le point 3 doit etre PLUS BAS que la ligne de fond lointaine (ligne de service loin)"
    if i == 3 and not (p[0] > pts[2][0] + 20 and abs(p[1] - pts[2][1]) < 0.08 * fh):
        return "Le point 4 doit etre a DROITE du point 3, sur la meme ligne de service loin"
    if i == 4 and p[1] <= pts[2][1] + 20:
        return "Les points 5-6 sont sur la ligne de service PROCHE : plus bas que le filet"
    if i == 5 and not (p[0] > pts[4][0] + 20 and abs(p[1] - pts[4][1]) < 0.08 * fh):
        return "Le point 6 doit etre a DROITE du point 5, sur la meme ligne de service proche"
    return None


def _draw_schema(img, n_done, x0=10, y0=44, w=150, h=300):
    """Petit schema du court avec les 6 points a cliquer (le prochain en jaune, les faits en vert)."""
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (30, 30, 30), -1)
    sx, sy = (w - 30) / COURT_W, (h - 30) / COURT_L
    P = lambda X, Y: (int(x0 + 15 + X * sx), int(y0 + 15 + Y * sy))
    col = (200, 200, 200)
    cv2.rectangle(img, P(0, 0), P(COURT_W, COURT_L), col, 1)
    cv2.line(img, P(SX0, 0), P(SX0, COURT_L), col, 1)
    cv2.line(img, P(SX1, 0), P(SX1, COURT_L), col, 1)
    cv2.line(img, P(SX0, SY_FAR), P(SX1, SY_FAR), col, 1)
    cv2.line(img, P(SX0, SY_NEAR), P(SX1, SY_NEAR), col, 1)
    cv2.line(img, P(COURT_W / 2, SY_FAR), P(COURT_W / 2, SY_NEAR), col, 1)
    cv2.line(img, P(-0.5, COURT_L / 2), P(COURT_W + 0.5, COURT_L / 2), (255, 255, 255), 2)
    cv2.putText(img, "FILET", (x0 + 4, y0 + h // 2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(img, "LOIN (en haut de l'image)", (x0 + 2, y0 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
    for i, (X, Y) in enumerate(CALIB_WORLD):
        c = (0, 220, 0) if i < n_done else ((0, 255, 255) if i == n_done else (120, 120, 120))
        cv2.circle(img, P(X, Y), 7 if i == n_done else 5, c, -1)
        cv2.putText(img, str(i + 1), (P(X, Y)[0] + 8, P(X, Y)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)


def pick_calibration(frame):
    """Clique les 6 points (schema a gauche, loupe x4 a droite). Apercu : ENTREE = valider, R = refaire, Z = annuler le dernier."""
    fh, fw = frame.shape[:2]
    s = min(1.0, 1280 / fw)
    base = cv2.resize(frame, None, fx=s, fy=s)
    bh = base.shape[0]
    pts, mouse, warn = [], [0, 0], ["", 0.0]
    win = "Calibration du court - clique les 6 points (Z = annuler, ESC = quitter)"
    cv2.namedWindow(win)

    def on_mouse(event, x, y, flags, param):
        mouse[0], mouse[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 6:
            err = _click_ok(pts, (x, y), bh)
            if err:
                warn[0], warn[1] = "Clic ignore : " + err, time.time() + 4.0
            else:
                pts.append((x, y))
    cv2.setMouseCallback(win, on_mouse)
    validated = False
    while not validated:
        img = base.copy()
        valid = False
        for i, p in enumerate(pts):
            cv2.circle(img, p, 4, (0, 0, 255), -1)
            cv2.putText(img, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        if len(pts) < 6:
            cv2.putText(img, CALIB_LABELS[len(pts)], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            mx, my = mouse                                   # loupe x4 autour du curseur
            r = 24
            x0, y0 = max(0, mx - r), max(0, my - r)
            patch = base[y0:y0 + 2 * r, x0:x0 + 2 * r]
            if patch.size:
                big = cv2.resize(patch, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                cx, cy = big.shape[1] // 2, big.shape[0] // 2
                cv2.line(big, (cx - 14, cy), (cx + 14, cy), (0, 0, 255), 1)
                cv2.line(big, (cx, cy - 14), (cx, cy + 14), (0, 0, 255), 1)
                bx = img.shape[1] - big.shape[1] - 10
                img[44:44 + big.shape[0], bx:bx + big.shape[1]] = big
                cv2.rectangle(img, (bx, 44), (bx + big.shape[1], 44 + big.shape[0]), (255, 255, 255), 1)
        else:
            try:
                H_, _, err = calib_from_points([(x / s, y / s) for x, y in pts])
                Hs = np.diag([s, s, 1.0]) @ np.linalg.inv(H_)       # metres -> pixels affiches
                draw_court_lines(img, np.linalg.inv(Hs), th=2)
                valid = err < 8.0
                txt = (f"Les lignes cyan collent aux lignes ?  ENTREE = oui   R = refaire   (erreur {err:.1f} px)" if valid
                       else f"POINTS FAUX (erreur {err:.0f} px) : appuie sur R et clique les points sur les LIGNES du court")
            except Exception:
                txt = "Points incoherents : appuie sur R pour refaire"
            cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255) if valid else (80, 80, 255), 2, cv2.LINE_AA)
        _draw_schema(img, len(pts))
        if time.time() < warn[1]:
            cv2.rectangle(img, (170, 44), (img.shape[1] - 230, 78), (0, 0, 120), -1)
            cv2.putText(img, warn[0][:110], (178, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit("Annule.")
        if k in (ord("z"), ord("Z")) and pts:
            pts.pop()
        if k in (ord("r"), ord("R")):
            pts.clear()
        if k in (13, 10) and len(pts) == 6 and valid:
            validated = True
    cv2.destroyWindow(win)
    res = [(round(x / s), round(y / s)) for x, y in pts]
    print("\nCalibration choisie -> colle cette ligne dans le fichier pour ne plus cliquer :")
    print(f"CALIB_POINTS = {res}\n")
    return res


# ---------------------------------------------------------------
# FONCTION 1b : zone de detection autour du court (recadrage)
# Marges : en haut (balle en l'air, joueur lointain), a droite/gauche, en bas (pieds du joueur proche)
# ---------------------------------------------------------------
def court_crop(corners, w, h):
    xs, ys = corners[:, 0], corners[:, 1]
    cw, ch = xs.max() - xs.min(), ys.max() - ys.min()
    x0 = int(max(0, xs.min() - 0.08 * cw))
    x1 = int(min(w, xs.max() + 0.08 * cw))
    y0 = int(max(0, ys.min() - 0.40 * ch))
    y1 = int(min(h, ys.max() + 0.15 * ch))
    return x0, y0, x1, y1


# ---------------------------------------------------------------
# FONCTION 2 : convertir un point pixel en position reelle (metres)
# ---------------------------------------------------------------
def to_world(H, pt):
    p = H @ np.array([pt[0], pt[1], 1.0])
    if abs(p[2]) < 1e-9 or p[2] < 0:
        return 1e5, 1e5                            # point au-dela de l'horizon du court : hors zone
    return p[0] / p[2], p[1] / p[2]


def make_height_checker(H):
    """Hauteur attendue (pixels) d'un adulte de 1.80 m debout au point (x, y) du sol, deduite de l'homographie."""
    Hi = np.linalg.inv(H)

    def ratio(world, box):
        q0 = Hi @ np.array([world[0], world[1], 1.0])
        q1 = Hi @ np.array([world[0] + 1.0, world[1], 1.0])
        scale = np.hypot(q1[0] / q1[2] - q0[0] / q0[2], q1[1] / q1[2] - q0[1] / q0[2])   # pixels par metre ici
        expected = 1.8 * 0.9 * scale
        return (box[3] - box[1]) / max(expected, 1.0)
    return ratio


def in_ball_zone(wx, wy):
    """La balle (au sol ou en l'air) doit rester au-dessus du court : pas dans les gradins, le cadran, les panneaux."""
    return -BALL_X_MARGIN < wx < COURT_W + BALL_X_MARGIN and -30.0 < wy < COURT_L + BALL_Y_MAX_MARGIN


# ---------------------------------------------------------------
# FONCTION 2b : reconnaitre la VUE LARGE du court
# On memorise la couleur du court sur la 1ere image, puis on mesure,
# pour chaque image, la part du polygone du court qui a cette couleur.
# Gros plan / public / replay => part faible => analyse en pause.
# ---------------------------------------------------------------
SMALL = 0.25                                       # echelle de calcul (rapide)


def make_court_checker(first_frame, corners):
    small = cv2.resize(first_frame, None, fx=SMALL, fy=SMALL)
    mask = np.zeros(small.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, (corners * SMALL).astype(np.int32), 255)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.int16)
    ref = np.median(hsv[mask > 0], axis=0)         # couleur mediane du court (H, S, V)

    def ratio(frame):
        sm = cv2.resize(frame, None, fx=SMALL, fy=SMALL)
        hv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV).astype(np.int16)
        dh = np.abs(hv[..., 0] - ref[0])
        dh = np.minimum(dh, 180 - dh)              # teinte circulaire
        ds = np.abs(hv[..., 1] - ref[1])
        dv = np.abs(hv[..., 2] - ref[2])
        match = (dh < 12) & (ds < 70) & (dv < 90)
        return float(match[mask > 0].mean())

    return ratio


# ---------------------------------------------------------------
# FONCTION 3 : mini-carte du court vue de dessus (joueurs + balle)
# ---------------------------------------------------------------
# Couleurs (RGB) : memes teintes que les etiquettes des joueurs
C_FAR, C_NEAR = (255, 200, 0), (0, 120, 255)
C_TXT, C_DIM, C_BALL, C_BAD = (245, 245, 245), (160, 170, 185), (255, 235, 59), (255, 110, 90)
_FONTS = {}


def _font(size, bold=True):
    size = int(max(8, size))
    key = (size, bold)
    if key in _FONTS:
        return _FONTS[key]
    names = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc"] if bold else
             ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Helvetica.ttc"])
    f = None
    for n in names:
        try:
            f = ImageFont.truetype(n, size)
            break
        except Exception:
            continue
    _FONTS[key] = f if f is not None else ImageFont.load_default()
    return _FONTS[key]


def _ascii(t):
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()


def _to_pil(roi_bgr):
    return Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")


def _from_pil(img):
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _card(size, radius, fill=(12, 16, 24, 205), outline=(255, 255, 255, 40)):
    """Un fond arrondi semi-transparent."""
    lay = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(lay).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius, fill=fill, outline=outline, width=1)
    return lay


# ---------------------------------------------------------------
# FONCTION 3 : mini-carte du court (vue de dessus, lignes reelles, balle + trainee)
# Le joueur "loin" est en haut, comme dans la video (MINIMAP_FLIP = True pour inverser)
# ---------------------------------------------------------------
def draw_minimap(frame, positions, ball_world, ball_trail_w=(), impacts=()):
    if Image is None:
        return _draw_minimap_basic(frame, positions, ball_world)
    fh, fw = frame.shape[:2]
    ss = 3                                         # dessine 3x plus grand puis reduit = bords lisses
    ch = int(MINIMAP_HEIGHT * fh)
    cw = int(ch * COURT_W / COURT_L)
    pad = max(10, int(ch * 0.07))
    top_bar = max(18, int(ch * 0.09))
    W, Hh = cw + 2 * pad, ch + 2 * pad + top_bar
    sc = ch * ss / COURT_L                         # pixels (grande image) par metre
    big = Image.new("RGBA", (W * ss, Hh * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    d.rounded_rectangle((0, 0, W * ss - 1, Hh * ss - 1), 14 * ss, fill=(12, 16, 24, 225), outline=(255, 255, 255, 60), width=ss)
    ox, oy = pad * ss, (pad + top_bar) * ss

    def P(x, y):                                   # metres -> pixels de la carte
        if MINIMAP_FLIP:
            x, y = COURT_W - x, COURT_L - y
        return ox + x * sc, oy + y * sc

    d.rectangle((*P(0, 0), *P(COURT_W, COURT_L)) if not MINIMAP_FLIP else (*P(COURT_W, COURT_L), *P(0, 0)),
                fill=(30, 94, 150, 255))           # surface bleue
    lw = max(2, int(1.6 * ss))
    white = (255, 255, 255, 235)
    s_in, net_y, svc = 1.37, COURT_L / 2, 6.40

    def line(a, b_, w_=lw):
        d.line([P(*a), P(*b_)], fill=white, width=w_)
    line((0, 0), (COURT_W, 0)); line((0, COURT_L), (COURT_W, COURT_L))      # fonds
    line((0, 0), (0, COURT_L)); line((COURT_W, 0), (COURT_W, COURT_L))      # doubles
    line((s_in, 0), (s_in, COURT_L)); line((COURT_W - s_in, 0), (COURT_W - s_in, COURT_L))   # simples
    line((s_in, net_y - svc), (COURT_W - s_in, net_y - svc))                # lignes de service
    line((s_in, net_y + svc), (COURT_W - s_in, net_y + svc))
    line((COURT_W / 2, net_y - svc), (COURT_W / 2, net_y + svc))            # ligne centrale
    d.line([P(-0.5, net_y), P(COURT_W + 0.5, net_y)], fill=(230, 230, 230, 255), width=int(3.2 * ss))   # filet

    far_lbl, near_lbl = ("PROCHE", "LOIN") if MINIMAP_FLIP else ("LOIN", "PROCHE")
    f_s = _font(max(9, ch * 0.055) * ss)
    d.text((W * ss / 2, (pad + top_bar / 2) * ss), "TERRAIN", font=_font(max(10, ch * 0.065) * ss),
           fill=C_TXT, anchor="mm")
    d.text((ox + 4 * ss, oy + 3 * ss), far_lbl, font=f_s, fill=(255, 255, 255, 150), anchor="la")
    d.text((ox + 4 * ss, oy + ch * ss - 3 * ss), near_lbl, font=f_s, fill=(255, 255, 255, 150), anchor="ld")

    tr = [t for t in ball_trail_w if t is not None]
    for i in range(1, len(tr)):                    # trainee de la balle
        a_ = int(60 + 150 * i / len(tr))
        d.line([P(*tr[i - 1]), P(*tr[i])], fill=(255, 235, 59, a_), width=int(2 * ss))
    for side, pos in positions.items():            # joueurs : point + halo
        if pos is None:
            continue
        x, y = P(np.clip(pos[0], 0, COURT_W), np.clip(pos[1], 0, COURT_L))
        r = max(4.5, ch * 0.032) * ss
        col = C_FAR if side == "far" else C_NEAR
        d.ellipse((x - r * 1.9, y - r * 1.9, x + r * 1.9, y + r * 1.9), fill=(*col, 70))
        d.ellipse((x - r, y - r, x + r, y + r), fill=(*col, 255), outline=(255, 255, 255, 255), width=int(1.5 * ss))
    for e in impacts:                              # impacts des derniers rebonds juges
        x, y = P(np.clip(e["world"][0], -0.8, COURT_W + 0.8), np.clip(e["world"][1], -1.5, COURT_L + 1.5))
        r = max(3.5, ch * 0.026) * ss
        c = (60, 220, 90) if e["verdict"] == "IN" else (255, 70, 60)
        d.line([(x - r, y - r), (x + r, y + r)], fill=(*c, 255), width=int(2.2 * ss))
        d.line([(x - r, y + r), (x + r, y - r)], fill=(*c, 255), width=int(2.2 * ss))
    if ball_world is not None:
        x, y = P(np.clip(ball_world[0], 0, COURT_W), np.clip(ball_world[1], 0, COURT_L))
        r = max(3, ch * 0.02) * ss
        d.ellipse((x - r, y - r, x + r, y + r), fill=(*C_BALL, 255), outline=(0, 0, 0, 255), width=int(1.2 * ss))
    small = big.resize((W, Hh), Image.LANCZOS)
    x0 = fw - W - max(10, int(0.012 * fw))
    y0 = int(np.clip(MINIMAP_Y * fh, 6, fh - Hh - 6))
    roi = frame[y0:y0 + Hh, x0:x0 + W]
    base = _to_pil(roi)
    base.alpha_composite(small)
    frame[y0:y0 + Hh, x0:x0 + W] = _from_pil(base)


def _draw_minimap_basic(frame, positions, ball_world):     # secours si Pillow n'est pas installe
    sc = 14
    mw, mh = int(COURT_W * sc), int(COURT_L * sc)
    mm = np.full((mh, mw, 3), (60, 120, 60), np.uint8)
    cv2.rectangle(mm, (0, 0), (mw - 1, mh - 1), (255, 255, 255), 2)
    cv2.line(mm, (0, mh // 2), (mw, mh // 2), (255, 255, 255), 2)
    for side, pos in positions.items():
        if pos is not None:
            cv2.circle(mm, (int(np.clip(pos[0], 0, COURT_W) * sc), int(np.clip(pos[1], 0, COURT_L) * sc)), 6,
                       (0, 200, 255) if side == "far" else (255, 120, 0), -1)
    if ball_world is not None:
        cv2.circle(mm, (int(np.clip(ball_world[0], 0, COURT_W) * sc), int(np.clip(ball_world[1], 0, COURT_L) * sc)), 3, (0, 255, 255), -1)
    fh, fw = frame.shape[:2]
    frame[fh - mh - 10: fh - 10, fw - mw - 10: fw - 10] = mm


# ---------------------------------------------------------------
# FONCTION 3b : panneau d'informations (haut-gauche) : net, lisible, avec barre de possession
# ---------------------------------------------------------------
def draw_hud(frame, d_, top=0):
    fh, fw = frame.shape[:2]
    if Image is None:
        return _draw_hud_basic(frame, d_)
    u = max(0.6, fw / 1920)
    f_lab, f_val, f_small = _font(21 * u, True), _font(30 * u, True), _font(20 * u, False)
    pad, rowh = int(20 * u), int(50 * u)
    lab_w = int(190 * u)
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))

    def tw(t, f):
        return tmp.textlength(t, font=f)
    rows = [
        ("VITESSE MAX", [("Loin ", C_DIM), (f"{d_['vmax_far']:.1f}", C_FAR), ("   Proche ", C_DIM),
                         (f"{d_['vmax_near']:.1f}", C_NEAR), (" km/h", C_DIM)]),
        ("BALLE", [(f"~{d_['ball_kmh']:.0f} km/h", C_BALL), (f"   max ~{d_['ball_max']:.0f}", C_DIM)]),
        ("ÉCHANGE", [(f"{d_['hits']} frappes", C_TXT), (f"   dernière : {d_['last']}", C_DIM)]),
        ("POSSESSION", None),
        ("RYTHME", [(f"{d_['rhythm']:.1f} s" if d_['rhythm'] > 0 else "—", C_TXT), (" par frappe", C_DIM)]),
        ("VAR", [("IN ", C_DIM), (str(d_['var_in']), (60, 220, 90)), ("   OUT ", C_DIM), (str(d_['var_out']), C_BAD),
                 (f"   dernier : {d_['var_last']}", C_DIM)]),
        ("FAUTES", [("Loin ", C_DIM), (str(d_['out_far']), C_BAD), ("   Proche ", C_DIM), (str(d_['out_near']), C_BAD),
                    ("   balles OUT", C_DIM)]),
    ]
    bar_w = int(400 * u)
    content_w = max(sum(tw(t, f_val) for t, _ in segs) for lab, segs in rows if segs)
    content_w = max(content_w, bar_w)
    pw = int(pad * 2 + lab_w + content_w)
    foot_h = int(34 * u)
    ph = pad + rowh * len(rows) + foot_h + int(pad * 0.4)
    ph = min(ph, fh - 12)
    x0, y0 = 10, 10 + top
    ph = max(40, min(ph, fh - y0 - 6))
    roi = frame[y0:y0 + ph, x0:x0 + pw]
    base = _to_pil(roi)
    base.alpha_composite(_card((pw, ph), int(14 * u)))
    dr = ImageDraw.Draw(base)
    dr.rounded_rectangle((0, int(14 * u), int(5 * u), ph - int(14 * u)), 2, fill=(255, 200, 0, 255))   # liseré
    y = pad
    for lab, segs in rows:
        cy = y + rowh // 2 - int(4 * u)
        dr.text((pad + int(8 * u), cy), lab, font=f_lab, fill=C_DIM, anchor="lm")
        x = pad + lab_w
        if segs is None:                                                         # barre de possession
            bh = int(26 * u)
            by = cy - bh // 2
            pf, pn = d_["poss_far"], d_["poss_near"]
            if d_["poss_ok"]:
                wf = int(bar_w * pf / 100)
                dr.rounded_rectangle((x, by, x + bar_w, by + bh), bh // 2, fill=(*C_NEAR, 255))
                if wf > 0:
                    dr.rounded_rectangle((x, by, x + max(wf, bh), by + bh), bh // 2, fill=(*C_FAR, 255))
                    dr.rectangle((x + max(wf, bh) - bh // 2, by, x + max(wf, bh), by + bh), fill=(*C_FAR, 255))
                    if wf < bar_w - bh:
                        dr.rectangle((x + wf, by, x + min(wf + bh // 2, bar_w - bh // 2), by + bh), fill=(*C_NEAR, 255))
                dr.text((x + int(10 * u), cy), f"Loin {pf:.0f}%", font=f_small, fill=(20, 20, 20), anchor="lm")
                dr.text((x + bar_w - int(10 * u), cy), f"Proche {pn:.0f}%", font=f_small, fill=(255, 255, 255), anchor="rm")
            else:
                dr.rounded_rectangle((x, by, x + bar_w, by + bh), bh // 2, fill=(60, 68, 82, 255))
                dr.text((x + bar_w // 2, cy), "en attente de la balle", font=f_small, fill=C_DIM, anchor="mm")
        else:
            for t, col in segs:
                f_use = f_small if col == C_DIM else f_val
                dr.text((x, cy), t, font=f_use, fill=col, anchor="lm")
                x += tw(t, f_use)
        y += rowh
    q = d_["tracked"]
    col = C_DIM if q >= 25 else C_BAD
    dr.text((pad + int(8 * u), y + int(8 * u)), f"Balle suivie sur {q:.0f}% des images", font=f_small, fill=col, anchor="lm")
    frame[y0:y0 + ph, x0:x0 + pw] = _from_pil(base)
    return pw


def _draw_hud_basic(frame, d_):                    # secours sans Pillow
    fs = max(0.6, frame.shape[1] / 1920 * 0.9)
    lines = [f"Vmax loin {d_['vmax_far']:.1f} | proche {d_['vmax_near']:.1f} km/h",
             f"Balle ~{d_['ball_kmh']:.0f} km/h", f"Echange {d_['hits']} frappes",
             (f"Possession loin {d_['poss_far']:.0f}% / proche {d_['poss_near']:.0f}%" if d_["poss_ok"] else "Possession : en attente"),
             f"VAR IN {d_['var_in']} / OUT {d_['var_out']}"]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (20, int(40 + i * 38 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, t, (20, int(40 + i * 38 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------
# FONCTION 4 : rectangle au-dessus de la tete avec la vitesse du joueur
# La taille du texte s'adapte a la resolution de la video (text_scale)
# ---------------------------------------------------------------
def draw_player_marker(frame, box, color, speed_kmh, text_scale):
    x1, y1, x2, y2 = [int(v) for v in box]
    cx = (x1 + x2) // 2
    text = "-- km/h" if speed_kmh is None else f"{speed_kmh:.1f} km/h"
    thick = max(1, int(round(text_scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thick)
    pad = max(4, int(6 * text_scale))
    bx1, bx2 = cx - tw // 2 - pad, cx + tw // 2 + pad
    by2 = y1 - 6
    by1 = by2 - th - 2 * pad
    if by1 < 0:
        by1 = 0
        by2 = th + 2 * pad
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
    text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
    cv2.putText(frame, text, (bx1 + pad, by2 - pad - base // 2),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, thick, cv2.LINE_AA)


# ---------------------------------------------------------------
# FONCTION 5 : choisir la bonne balle parmi les candidats
# ---------------------------------------------------------------
def select_ball(cands, last_pos, max_jump):
    if not cands:
        return None
    if last_pos is None:
        best = max(cands, key=lambda c: c[2])
        return (best[0], best[1]) if best[2] >= BALL_START_CONF else None
    nearest = min(cands, key=lambda c: np.hypot(c[0] - last_pos[0], c[1] - last_pos[1]))
    if np.hypot(nearest[0] - last_pos[0], nearest[1] - last_pos[1]) <= max_jump:
        return (nearest[0], nearest[1])
    return None                                    # trop loin : probablement une fausse detection


# ---------------------------------------------------------------
# FONCTION 6 : dessiner la balle et sa traînee
# ---------------------------------------------------------------
def draw_ball(frame, trail, show_head=True):
    pts = list(trail)
    for i in range(1, len(pts)):
        thickness = max(1, int(1 + 3 * i / len(pts)))
        if np.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) > 0.25 * frame.shape[1]:
            continue                               # pas de trait geant entre deux points sans rapport
        cv2.line(frame, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                 (int(pts[i][0]), int(pts[i][1])), (0, 255, 255), thickness)
    if pts and show_head:
        cv2.circle(frame, (int(pts[-1][0]), int(pts[-1][1])), 6, (0, 255, 255), 2)


# ---------------------------------------------------------------
# FONCTION 6a : vitesse d'un joueur, robuste
#   - position lissee (les pieds detectes tremblent), et saut impossible = erreur de detection ignoree
#   - vitesse = pente d'une droite ajustee sur les derniers points (pas seulement 2 points)
#   - vitesse affichee lissee ; la vitesse max n'est mise a jour que quand la mesure est stable
# ---------------------------------------------------------------
def update_speed(history, cur_speed, max_speed, side, frame_idx, world, eff_fps):
    h = history[side]
    if h:
        f_prev, x_prev, y_prev = h[-1]
        gap = max(1, frame_idx - f_prev)
        step = np.hypot(world[0] - x_prev, world[1] - y_prev)
        if step > 1.6 * MAX_HUMAN_SPEED * gap / eff_fps + 0.4:
            return                                 # saut impossible pour un humain : detection erronee, on l'ignore
        world = (SPEED_POS_EMA * world[0] + (1 - SPEED_POS_EMA) * x_prev,
                 SPEED_POS_EMA * world[1] + (1 - SPEED_POS_EMA) * y_prev)
    h.append((frame_idx, world[0], world[1]))
    if len(h) < 4:
        return
    t = np.array([p[0] for p in h], float) / eff_fps
    vx = np.polyfit(t, [p[1] for p in h], 1)[0]
    vy = np.polyfit(t, [p[2] for p in h], 1)[0]
    v = float(np.hypot(vx, vy))
    if v > MAX_HUMAN_SPEED:
        return
    cur_speed[side] = v if cur_speed[side] <= 0 else SPEED_EMA * v + (1 - SPEED_EMA) * cur_speed[side]
    if len(h) > WINDOW:
        max_speed[side] = max(max_speed[side], cur_speed[side])


# ---------------------------------------------------------------
# FONCTION 6b : verrouiller les 2 joueurs
# Probleme regle : quand une personne (ramasseur, arbitre, photographe...) entre sur le court,
# elle ne devient plus "joueur". Chaque joueur est SUIVI : a chaque image on garde la personne
# la plus proche de sa derniere position (dans la distance qu'un humain peut parcourir).
# Une personne qui apparait ailleurs est ignoree tant que le joueur est suivi.
# Entree : candidats [(aire, pieds, (x_m, y_m), boite)], derniere position de chaque cote
# Sortie : {"far": candidat ou None, "near": candidat ou None}
# ---------------------------------------------------------------
def assign_players(cands, last_pos, last_seen, frame_idx, eff_fps):
    best = {"far": None, "near": None}
    taken = set()
    lost_max = LOST_SECONDS * eff_fps

    def locked(side):
        return last_pos[side] is not None and frame_idx - last_seen[side] <= lost_max

    # 1) joueurs deja suivis : la personne la plus proche de leur derniere position
    options = []
    for side in ("far", "near"):
        if not locked(side):
            continue
        gap = frame_idx - last_seen[side]
        reach = min(0.8 * MAX_HUMAN_SPEED * gap / eff_fps + 0.8, 8.0)   # distance max parcourue (m) + marge d'erreur
        for i, c in enumerate(cands):
            d = np.hypot(c[2][0] - last_pos[side][0], c[2][1] - last_pos[side][1])
            if d <= reach:
                options.append((d, side, i))
    for d, side, i in sorted(options):
        if best[side] is None and i not in taken:
            best[side] = cands[i]
            taken.add(i)

    # 2) cote sans joueur suivi (debut de video, apres une coupure) : le plus grand cadre libre
    #    de la bonne moitie du court, de preference a l'interieur des lignes
    for side in ("far", "near"):
        if best[side] is not None or locked(side):
            continue            # un joueur suivi mais non vu un instant : on attend, on ne prend personne d'autre
        pool = []
        for i, c in enumerate(cands):
            if i in taken or (c[2][1] < COURT_L / 2) != (side == "far"):
                continue
            if len(c) > 4 and c[4] < PERSON_CONF:
                continue                           # faible confiance : acceptable pour SUIVRE un joueur, pas pour le choisir
            sy = START_Y_FAR if side == "far" else START_Y_NEAR
            if not (-START_X <= c[2][0] <= COURT_W + START_X and -sy <= c[2][1] <= COURT_L + sy):
                continue                           # juge de ligne, ramasseur, arbitre : on ne les choisit jamais comme joueur
            dedans = 0 <= c[2][0] <= COURT_W and 0 <= c[2][1] <= COURT_L
            pool.append((c[0] * (1.0 if dedans else 0.4), i))
        if pool:
            _, i = max(pool)
            best[side] = cands[i]
            taken.add(i)
    return best


# ---------------------------------------------------------------
# FONCTION 6c : fenetre de lecture avec boutons (pause, retour, avance, zoom)
#   Boutons en bas de la fenetre, ou clavier :
#     ESPACE / p : pause - lecture      a / d (ou fleches) : image -1 / +1
#     j / l : -10 / +10 images          molette ou + / - : zoom      glisser : deplacer
#     0 ou r : zoom normal              q : quitter
#   En pause, l'analyse s'arrete vraiment. A la reprise, les images deja vues sont rejouees,
#   puis le calcul continue. A la fin de la video, la fenetre reste ouverte pour tout revoir.
# ---------------------------------------------------------------
class Viewer:
    WIN = "Tennis Analyzer v9.3 (loupe VAR)"
    TB_H = 0                                       # plus de barre de boutons : commandes au clavier / souris seulement
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    BUTTONS = [("<<", "back10"), ("<", "back1"), ("PAUSE", "play"), (">", "fwd1"), (">>", "fwd10"),
               ("ZOOM +", "zin"), ("ZOOM -", "zout"), ("RESET", "reset")]

    def __init__(self, w, h, fps, enabled=True):
        self.enabled = enabled
        self.w, self.h, self.fps = w, h, fps
        self.disp_w = min(max(w, 960), 1280)       # largeur affichee (agrandit les petites videos)
        self.disp_h = int(round(h * self.disp_w / w))
        self.buf = deque(maxlen=BUFFER_FRAMES)     # (id, image JPEG) des images annotees
        self.uid = 0
        self.view_idx = -1                         # image affichee dans buf
        self.paused = False
        self.quit = False
        self.finished = False
        self.zoom = 1.0
        self.cx, self.cy = w / 2, h / 2            # centre de la zone zoomee (pixels de la video)
        self.dirty = True
        self.cache = (None, None)                  # (id, image decodee)
        self.drag = None
        self.var_events = []                       # decisions VAR : {uid, px, verdict, d, limite, img}
        self.var_i = -1
        self.var_mode = False
        self.var_static = None                     # image enregistree (si l'image est sortie de la memoire)
        self.var_btn = None
        self.dec_btn = None
        self.msg = ("", 0.0)
        self.mouse = (0, 0)
        self.manual_cb = None                      # fonction(frame_idx, (x,y)) -> decision, fournie par main()
        self.fidx = {}                             # uid -> numero de l'image video
        self.in_btn = self.out_btn = None
        self.revealed = False                      # le verdict IN/OUT est-il affiche sur le bouton DECISION ?
        self.hud_w = 360                           # largeur (pixels video) du panneau de stats, pour aligner l'en-tete
        if enabled:
            cv2.namedWindow(self.WIN)
            cv2.setMouseCallback(self.WIN, self.on_mouse)
            print("Rectangle en haut a gauche : bouton VAR (arrete la video) puis bouton DECISION (IN / OUT) | "
                  "ESPACE pause | a/d ou fleches : -1/+1 image | j/l : -10/+10 | molette ou +/- : zoom | q : quitter")

    def add_var_event(self, ev, img_path):
        """A appeler juste apres push() de l'image ou la decision a ete prise."""
        if not self.enabled:
            return
        self.var_events.append({"uid": self.uid, "px": ev["px"], "verdict": ev["verdict"], "d": ev["d"],
                                "limite": ev["limite"], "img": img_path, "t": ev["t"], "unc": ev["unc"]})
        self.dirty = True

    # hauteur reservee en haut a gauche (en pixels de la video) pour l'en-tete VAR : le panneau de stats se place dessous
    @staticmethod
    def reserved_top(w):
        disp_w = min(max(w, 960), 1280)
        return int(round(112 * w / disp_w))

    def _show_event(self, i):
        e = self.var_events[i]
        self.var_i = i
        idx = e["uid"] - self.buf[0][0] if self.buf else -1
        if 0 <= idx < len(self.buf):
            self.view_idx, self.var_static = idx, None
            self.zoom = 3.0
            self.cx, self.cy = e["px"][0], e["px"][1]          # zoom centre sur le point d'impact
        else:                                      # l'image n'est plus en memoire : on montre l'image enregistree
            self.var_static = cv2.imread(e["img"]) if e.get("img") else None
            self.zoom, self.cx, self.cy = 1.0, self.w / 2, self.h / 2

    def _enter_var(self):
        """Bouton VAR : arrete la video. S'il y a une decision, on se place sur la plus recente."""
        self.var_mode, self.paused, self.revealed = True, True, False
        if self.var_events:
            self._show_event(len(self.var_events) - 1)
        else:
            self.var_i = -1

    def _leave_var(self):
        self.var_mode, self.var_static, self.revealed = False, None, False
        self.dec_btn = None
        self.zoom, self.cx, self.cy = 1.0, self.w / 2, self.h / 2

    def _jump(self, verdict):
        """Boutons IN / OUT : arrete la video et va a la decision IN (ou OUT) suivante (la plus recente d'abord)."""
        idxs = [i for i, e in enumerate(self.var_events) if e["verdict"] == verdict]
        if not idxs:
            self.paused = True                     # rien a montrer : on arrete quand meme, et on le dit a l'ecran
            self.msg = (f"Aucune decision {verdict} : clique sur l'image", time.time() + 3.0)
            self.dirty = True
            return
        cur = self.var_i if (self.var_mode and self.var_i in idxs) else None
        nxt = idxs[-1] if cur is None else idxs[(idxs.index(cur) - 1) % len(idxs)]
        self.var_mode, self.paused, self.revealed = True, True, True
        self._show_event(nxt)

    def _manual_click(self, x, y):
        """Video arretee par VAR : un clic sur l'image = endroit ou la balle touche le sol -> IN / OUT."""
        if self.manual_cb is None or not self.buf or self.var_static is not None:
            return
        x0, y0, vw, vh = self.region()
        fx, fy = x0 + x / self.disp_w * vw, y0 + y / self.disp_h * vh
        uid = self.buf[self.view_idx][0]
        ev = self.manual_cb(self.fidx.get(uid, uid), (fx, fy))
        if ev is None:
            self.msg = ("Clic trop loin du terrain", time.time() + 2.5)
            return
        frame = self.current_frame()
        path = save_var_image(frame, ev, ev.get("n", 0) or int(time.time()) % 100000)
        self.var_events.append({"uid": uid, "px": ev["px"], "verdict": ev["verdict"], "d": ev["d"],
                                "limite": ev["limite"], "img": path, "t": ev["t"], "unc": ev["unc"]})
        self.var_i = len(self.var_events) - 1
        self.revealed = True
        self.dirty = True

    def _decision_click(self):
        """Bouton DECISION : 1er clic = montre IN / OUT ; clics suivants = decision precedente."""
        if not self.var_mode or not self.var_events:
            return
        if not self.revealed:
            self.revealed = True
        else:
            self._show_event((self.var_i - 1) % len(self.var_events))
            self.revealed = True

    # --- memoire des images ---
    def push(self, frame, frame_idx=None):
        if not self.enabled:
            return
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return
        following = self.view_idx >= len(self.buf) - 1          # on regarde la derniere image ?
        if len(self.buf) == self.buf.maxlen and not following:
            self.view_idx = max(0, self.view_idx - 1)          # la plus ancienne image vient de sortir
        self.uid += 1
        self.buf.append((self.uid, jpg))
        self.fidx[self.uid] = frame_idx if frame_idx is not None else self.uid
        if len(self.fidx) > len(self.buf) + 50:
            lo = self.buf[0][0]
            self.fidx = {k: v for k, v in self.fidx.items() if k >= lo}
        if following:
            self.view_idx = len(self.buf) - 1
            self.cache = (self.uid, frame)                      # deja decodee : pas besoin de la relire
        self.dirty = True

    def current_frame(self):
        uid, jpg = self.buf[self.view_idx]
        if self.cache[0] != uid:
            self.cache = (uid, cv2.imdecode(jpg, cv2.IMREAD_COLOR))
        return self.cache[1]

    # --- zoom ---
    def region(self):
        vw, vh = self.w / self.zoom, self.h / self.zoom
        x0 = min(max(self.cx - vw / 2, 0), self.w - vw)
        y0 = min(max(self.cy - vh / 2, 0), self.h - vh)
        self.cx, self.cy = x0 + vw / 2, y0 + vh / 2
        return x0, y0, vw, vh

    def zoom_at(self, mx, my, factor):
        x0, y0, vw, vh = self.region()
        fx, fy = x0 + mx / self.disp_w * vw, y0 + my / self.disp_h * vh   # point sous la souris
        self.zoom = min(8.0, max(1.0, self.zoom * factor))
        nvw, nvh = self.w / self.zoom, self.h / self.zoom
        self.cx = fx - mx / self.disp_w * nvw + nvw / 2                    # ce point reste sous la souris
        self.cy = fy - my / self.disp_h * nvh + nvh / 2
        self.dirty = True

    # --- affichage ---
    def render(self):
        if not self.enabled or not self.buf:
            return
        frame = self.current_frame()
        x0, y0, vw, vh = self.region()
        crop = frame[int(y0):int(y0 + vh), int(x0):int(x0 + vw)]
        interp = cv2.INTER_LINEAR if self.zoom == 1.0 else cv2.INTER_CUBIC
        img = cv2.resize(crop, (self.disp_w, self.disp_h), interpolation=interp)

        if self.var_static is not None:            # decision plus ancienne que la memoire : image enregistree
            st = self.var_static
            sc = min(self.disp_w / st.shape[1], self.disp_h / st.shape[0])
            st = cv2.resize(st, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            img = np.zeros((self.disp_h, self.disp_w, 3), np.uint8)
            oy, ox = (self.disp_h - st.shape[0]) // 2, (self.disp_w - st.shape[1]) // 2
            img[oy:oy + st.shape[0], ox:ox + st.shape[1]] = st

        if self.finished:                          # page de fin : plus de VAR, il n'existe que pendant la video
            self.var_btn = self.dec_btn = self.in_btn = self.out_btn = None
            self.var_mode = False
        else:
            self._draw_header(img)
        if self.var_mode:
            if 0 <= self.var_i < len(self.var_events):          # marque l'endroit juge
                e = self.var_events[self.var_i]
                sx0, sy0, svw, svh = self.region()
                mx = int((e["px"][0] - sx0) / svw * self.disp_w)
                my = int((e["px"][1] - sy0) / svh * self.disp_h)
                mc = (60, 220, 90) if e["verdict"] == "IN" else (60, 70, 255)
                cv2.drawMarker(img, (mx, my), (0, 0, 0), cv2.MARKER_CROSS, 26, 4)
                cv2.drawMarker(img, (mx, my), mc, cv2.MARKER_CROSS, 24, 2)
                cv2.circle(img, (mx, my), 14, mc, 2, cv2.LINE_AA)
            self._draw_loupe(img, frame)
            hint = "VAR : clique sur le point de rebond (a / d = image par image). Clic sur VAR = reprendre"
            (tw_, th_), _ = cv2.getTextSize(hint, self.FONT, 0.5, 1)
            cv2.rectangle(img, (6, self.disp_h - 14 - th_ - 8), (14 + tw_, self.disp_h - 4), (15, 18, 26), -1)
            cv2.putText(img, hint, (10, self.disp_h - 12), self.FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        canvas = img
        cv2.imshow(self.WIN, canvas)
        self.dirty = False

    def _draw_header(self, img):
        """Rectangle haut-gauche : [VAR] et, apres un clic sur VAR, [DECISION]. Les stats (score) sont dessous."""
        F = self.FONT
        x0, y0, bh = 10, 10, 46
        sc = self.disp_w / self.w
        card_w = max(int(self.hud_w * sc), 330)
        card_h = bh + 12 + 44
        cv2.rectangle(img, (x0, y0), (x0 + card_w, y0 + card_h), (18, 22, 30), -1)
        cv2.rectangle(img, (x0, y0), (x0 + card_w, y0 + card_h), (255, 200, 0), 1)
        # --- bouton VAR ---
        n = len(self.var_events)
        v1, v2 = (x0 + 6, y0 + 6), (x0 + 6 + 150, y0 + 6 + bh)
        self.var_btn = (v1[0], v1[1], v2[0], v2[1])
        col = (60, 70, 255) if not self.var_mode else (60, 190, 255)
        cv2.rectangle(img, v1, v2, col, -1)
        lab = "VAR  STOP" if not self.var_mode else "VAR  PLAY"
        cv2.putText(img, lab, (v1[0] + 12, v1[1] + 29), F, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if n:
            cv2.putText(img, str(n), (v2[0] - 20, v1[1] + 16), F, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        # --- boutons IN / OUT (sous VAR) : aller aux decisions IN ou OUT ---
        n_in = sum(e["verdict"] == "IN" for e in self.var_events)
        n_out = n - n_in
        by1 = v2[1] + 8
        hw = (card_w - 12 - 8) // 2
        for k, (lab, cnt, colr, name) in enumerate(((VAR_WORD_IN, n_in, (60, 170, 50), "in"), ("OUT", n_out, (50, 55, 230), "out"))):
            b1 = (x0 + 6 + k * (hw + 8), by1)
            b2 = (b1[0] + hw, by1 + 36)
            setattr(self, name + "_btn", (b1[0], b1[1], b2[0], b2[1]))
            cv2.rectangle(img, b1, b2, colr if cnt else (70, 74, 84), -1)
            cv2.putText(img, f"{lab}  ({cnt})", (b1[0] + 14, b1[1] + 26), F, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if time.time() < self.msg[1]:              # message temporaire (ex. aucune decision IN)
            cv2.rectangle(img, (v2[0] + 8, v1[1]), (x0 + card_w - 6, v2[1]), (40, 40, 120), -1)
            cv2.putText(img, self.msg[0], (v2[0] + 14, v1[1] + 29), F, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            self.dec_btn = None
            return
        # --- bouton DECISION (n'existe que quand la video est arretee par VAR) ---
        self.dec_btn = None
        if self.var_mode:
            d1, d2 = (v2[0] + 8, v1[1]), (min(x0 + card_w - 6, v2[0] + 8 + 230), v2[1])
            self.dec_btn = (d1[0], d1[1], d2[0], d2[1])
            if not self.var_events:
                cv2.rectangle(img, d1, d2, (70, 74, 84), -1)
                cv2.putText(img, "AUCUNE DECISION", (d1[0] + 10, d1[1] + 29), F, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
            elif not self.revealed:
                cv2.rectangle(img, d1, d2, (200, 120, 20), -1)
                cv2.putText(img, f"DECISION ? ({self.var_i + 1}/{n})", (d1[0] + 10, d1[1] + 29), F, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                e = self.var_events[self.var_i]
                ok = e["verdict"] == "IN"
                cv2.rectangle(img, d1, d2, (60, 170, 50) if ok else (50, 55, 230), -1)
                mot = VAR_WORD_IN if ok else "OUT"
                txt = f"{mot}  {abs(e['d']) * 100:.0f} cm" + (" ?" if e["limite"] else "")
                cv2.putText(img, txt, (d1[0] + 10, d1[1] + 31), F, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_loupe(self, img, frame):
        """Loupe a DROITE (video arretee par VAR) : agrandit les vrais pixels autour de la souris pour viser le point de rebond."""
        mx, my = self.mouse
        if not (0 <= mx < self.disp_w and 0 <= my < self.disp_h) or self.var_static is not None:
            return
        x0, y0, vw, vh = self.region()
        fx, fy = x0 + mx / self.disp_w * vw, y0 + my / self.disp_h * vh
        r = 12                                     # demi-cote (pixels video) de la zone agrandie
        ix, iy = int(round(fx)), int(round(fy))
        xa, ya = max(0, ix - r), max(0, iy - r)
        patch = frame[ya:iy + r, xa:ix + r]
        if patch.size == 0:
            return
        size = 200
        big = cv2.resize(patch, (size, size), interpolation=cv2.INTER_CUBIC)
        c = size // 2
        cv2.line(big, (c - 18, c), (c + 18, c), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(big, (c, c - 18), (c, c + 18), (0, 0, 255), 1, cv2.LINE_AA)
        bx, by = self.disp_w - size - 14, 14
        img[by:by + size, bx:bx + size] = big
        cv2.rectangle(img, (bx, by), (bx + size, by + size), (255, 255, 255), 2)
        cv2.putText(img, "LOUPE", (bx + 6, by + 18), self.FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # --- commandes ---
    def action(self, name):
        last = len(self.buf) - 1
        if name == "var":
            if self.var_mode:
                self._leave_var()                  # 2e clic : on reprend la lecture
                self.paused = False
            else:
                self._enter_var()
        elif name == "decision":
            self._decision_click()
        elif name in ("in", "out"):
            self._jump("IN" if name == "in" else "OUT")
        elif name == "play":
            if self.var_mode:
                self._leave_var()                  # on quitte la revue : la lecture reprend depuis la decision
                self.paused = False
            else:
                self.paused = not self.paused      # (en fin de video, PLAY rejoue depuis l'image affichee)
        elif name in ("back1", "back10", "fwd1", "fwd10"):
            step = 10 if name.endswith("10") else 1
            self.paused = True
            self.view_idx = max(0, min(last, self.view_idx + (step if name.startswith("fwd") else -step)))
        elif name == "zin":
            self.zoom_at(self.disp_w / 2, self.disp_h / 2, 1.25)
        elif name == "zout":
            self.zoom_at(self.disp_w / 2, self.disp_h / 2, 1 / 1.25)
        elif name == "reset":
            self.zoom, self.cx, self.cy = 1.0, self.w / 2, self.h / 2
            self.var_mode, self.var_static = False, None
        elif name == "quit":
            self.quit = True
        self.dirty = True

    def on_key(self, k):
        if k == -1:
            return
        if k in (2424832, 65361):                  # fleche gauche (Windows, Linux)
            return self.action("back1")
        if k in (2555904, 65363):                  # fleche droite
            return self.action("fwd1")
        c = chr(k & 0xFF).lower()
        table = {" ": "play", "p": "play", "a": "back1", ",": "back1", "d": "fwd1", ".": "fwd1",
                 "j": "back10", "l": "fwd10", "+": "zin", "=": "zin", "-": "zout", "_": "zout",
                 "0": "reset", "r": "reset", "v": "var", "i": "decision", "q": "quit"}
        if c in table:
            self.action(table[c])

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse = (x, y)
            if self.var_mode:
                self.dirty = True                  # la loupe suit la souris
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn, act in ((self.var_btn, "var"), (self.dec_btn, "decision"), (self.in_btn, "in"), (self.out_btn, "out")):
                if btn and btn[0] <= x <= btn[2] and btn[1] <= y <= btn[3]:
                    self.action(act)                       # clic sur VAR ou sur DECISION
                    return
            self.drag = (x, y, self.cx, self.cy)           # debut du deplacement (zone zoomee)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            x0m, y0m, cx0, cy0 = self.drag
            vw, vh = self.w / self.zoom, self.h / self.zoom
            self.cx = cx0 - (x - x0m) / self.disp_w * vw
            self.cy = cy0 - (y - y0m) / self.disp_h * vh
            self.dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            d0 = self.drag
            self.drag = None
            if d0 is not None and abs(x - d0[0]) <= 4 and abs(y - d0[1]) <= 4 and self.var_mode:
                self._manual_click(x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags) if hasattr(cv2, "getMouseWheelDelta") else (1 if flags > 0 else -1)
            my = min(y, self.disp_h - 1)
            self.zoom_at(x, my, 1.25 if delta > 0 else 1 / 1.25)

    def _wait_while_paused(self):
        while self.paused and not self.quit:
            if self.dirty:
                self.render()
            self.on_key(cv2.waitKeyEx(30))

    # --- a appeler apres chaque image traitee : affiche, ecoute, bloque si pause ---
    def tick(self):
        if not self.enabled:
            return True
        self.render()
        self.on_key(cv2.waitKeyEx(1))
        self._wait_while_paused()
        # apres une reprise : on rejoue les images deja traitees avant de continuer le calcul
        while not self.quit and not self.paused and self.view_idx < len(self.buf) - 1:
            self.view_idx += 1
            self.render()
            self.on_key(cv2.waitKeyEx(max(1, int(1000 / self.fps))))
            self._wait_while_paused()
        return not self.quit

    # --- a la fin de la video : la fenetre reste ouverte pour revoir et zoomer ---
    def review(self):
        if not self.enabled or not self.buf:
            return
        self.finished = True
        self.paused = True
        self.dirty = True
        print("\nFin de la video : tu peux encore reculer, avancer et zoomer. Appuie sur q pour fermer.")
        while not self.quit:
            if self.dirty:
                self.render()
            self.on_key(cv2.waitKeyEx(30 if self.paused else max(1, int(1000 / self.fps))))
            if not self.paused:                    # lecture : on avance jusqu'a la derniere image
                if self.view_idx < len(self.buf) - 1:
                    self.view_idx += 1
                    self.dirty = True
                else:
                    self.paused = True
                    self.dirty = True


# ---------------------------------------------------------------
# FONCTION 5b : balle par la couleur (jaune-vert fluo) + mouvement.
# Complete YOLO, qui rate souvent la balle (trop petite) : on cherche des petites taches
# jaune-vert qui BOUGENT. Les filtres (sur un joueur, immobile, saut trop grand) restent actifs.
# ---------------------------------------------------------------
def color_ball_candidates(region, prev_gray, off_x, off_y, w):
    gray = cv2.GaussianBlur(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    if prev_gray is None or prev_gray.shape != gray.shape:
        return [], gray
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    col = cv2.inRange(hsv, (26, 95, 150), (46, 255, 255))                    # jaune-vert vif
    motion = cv2.dilate((cv2.absdiff(gray, prev_gray) > 22).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
    mask = cv2.bitwise_and(col, motion)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out, amax = [], (0.009 * w) ** 2
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 3 or area > amax or max(bw, bh) > 1.9 * min(bw, bh) + 1:
            continue
        out.append((float(cents[i][0] + off_x), float(cents[i][1] + off_y), 0.30, "couleur"))
    return out, gray


# ---------------------------------------------------------------
# FONCTION 6d : possession + rapidite du jeu (echanges, frappes, rythme)
#   Frappe  = la balle fait demi-tour pres d'un joueur (zone elargie autour de sa boite)
#   Echange = suite de frappes ; il se termine quand la balle n'est plus vue pendant RALLY_TIMEOUT secondes
#   Possession = part du temps ou la balle est dans le camp de chaque joueur (pendant les echanges)
#   Rythme = temps moyen entre deux frappes ; plus c'est petit, plus le jeu est rapide
#   ATTENTION : tout depend du suivi de la balle (si elle n'est pas vue, les frappes sont ratees).
# ---------------------------------------------------------------
class RallyTracker:
    NOM = {"far": "loin", "near": "proche"}

    def __init__(self, eff_fps, width):
        self.fps, self.w = eff_fps, width
        self.track = deque(maxlen=3)               # 3 dernieres vraies detections (image, x, y)
        self.last_seen = -10 ** 9
        self.active = False
        self.start = 0
        self.last_hit = -10 ** 9
        self.hitters = []                          # camp de chaque frappe de l'echange en cours
        self.last_hitter = None
        self.rallies = []                          # echanges termines
        self.poss = {"far": 0.0, "near": 0.0}      # secondes de balle dans chaque camp
        self.hits_by = {"far": 0, "near": 0}
        self.gaps = []                             # secondes entre deux frappes d'un meme echange
        self.speeds = []                           # vitesses de balle pendant les echanges (m/s)
        self.n_frames = 0
        self.n_ball = 0
        self.faults = {"far": 0, "near": 0}        # services sans retour (faute probable ou ace), par serveur

    # --- appele a chaque image traitee ---
    def update(self, frame_idx, ball_px, ball_world, ball_speed, boxes):
        if self.active and frame_idx - self.last_seen > RALLY_TIMEOUT * self.fps:
            self.end()
        self.n_frames += 1
        if ball_px is not None:
            self.n_ball += 1
        if ball_px is None:
            return
        if self.track and frame_idx - self.track[-1][0] > 3:
            self.track.clear()                     # trou trop long : on ne relie pas deux trajectoires
        self.last_seen = frame_idx
        self.track.append((frame_idx, ball_px[0], ball_px[1]))
        if self.active and ball_world is not None:
            cote = "far" if ball_world[1] < COURT_L / 2 else "near"
            self.poss[cote] += 1.0 / self.fps
            if ball_speed > 0:
                self.speeds.append(ball_speed)
        self.detect_hit(boxes)

    def hitter_at(self, p, boxes):
        best, best_d = None, 1e9
        for side, box in boxes.items():
            if box is None:
                continue
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            ex = max(0.6 * bw, 0.025 * self.w)
            ey_top = max(0.5 * bh, 0.03 * self.w)
            ey_bot = 0.1 * bh
            if x1 - ex <= p[0] <= x2 + ex and y1 - ey_top <= p[1] <= y2 + ey_bot:
                d = np.hypot(p[0] - (x1 + x2) / 2, p[1] - (y1 + y2) / 2)
                if d < best_d:
                    best, best_d = side, d
        return best

    def detect_hit(self, boxes):
        if len(self.track) < 3:
            return
        (f0, x0, y0), (f1, x1, y1), (f2, x2, y2) = self.track
        vin, vout = (x1 - x0, y1 - y0), (x2 - x1, y2 - y1)
        nin, nout = np.hypot(*vin), np.hypot(*vout)
        if nin < HIT_MIN_MOVE * self.w or nout < HIT_MIN_MOVE * self.w:
            return
        if (vin[0] * vout[0] + vin[1] * vout[1]) / (nin * nout) > HIT_COS:
            return                                 # la balle ne fait pas demi-tour
        if f1 - self.last_hit < HIT_DEBOUNCE * self.fps:
            return
        side = self.hitter_at((x1, y1), boxes)
        if side is None:
            return                                 # demi-tour loin des joueurs (rebond, fausse detection)
        if self.hitters and self.hitters[-1] == side and f1 - self.last_hit < HIT_SAME_PLAYER * self.fps:
            return                                 # meme joueur deux fois de suite : doublon
        if not self.active:
            self.active, self.start, self.hitters = True, f1, []
        else:
            self.gaps.append((f1 - self.last_hit) / self.fps)
        self.hitters.append(side)
        self.hits_by[side] += 1
        self.last_hit, self.last_hitter = f1, side

    def end(self):
        if not self.active:
            return
        if len(self.hitters) == 1:
            self.faults[self.hitters[0]] += 1
        self.rallies.append({
            "debut_s": round(self.start / self.fps, 1),
            "duree_s": round((self.last_hit - self.start) / self.fps, 1),
            "frappes": len(self.hitters),
            "service": self.NOM[self.hitters[0]],
            "enchainement": "".join("L" if h == "far" else "P" for h in self.hitters),
        })
        self.active, self.hitters = False, []
        self.track.clear()

    def interrupt(self):                           # coupure de camera : l'echange est termine
        self.end()
        self.track.clear()

    # --- chiffres ---
    def possession(self):
        tot = self.poss["far"] + self.poss["near"]
        pf = 100 * self.poss["far"] / tot if tot else 0.0
        return (pf, 100 - pf) if tot else (0.0, 0.0)

    def rhythm(self):
        return float(np.mean(self.gaps)) if self.gaps else 0.0

    def live_data(self):
        pf, pn = self.possession()
        n = len(self.hitters) if self.active else 0
        return {"hits": n,
                "last": self.NOM[self.last_hitter] if (self.active and self.last_hitter) else "—",
                "poss_far": pf, "poss_near": pn, "poss_ok": (self.poss["far"] + self.poss["near"]) > 0.5,
                "rhythm": self.rhythm(), "f_far": self.faults["far"], "f_near": self.faults["near"],
                "n_rallies": len(self.rallies),
                "tracked": 100.0 * self.n_ball / max(1, self.n_frames)}

    def summary_frame(self, w, h, vmax_far, vmax_near, ball_max):
        img = np.full((h, w, 3), (35, 25, 20), np.uint8)
        pf, pn = self.possession()
        longs = [r for r in self.rallies if r["frappes"] >= 2]
        rh = self.rhythm()
        lignes = [
            ("RESULTATS DU MATCH ANALYSE", (0, 220, 255), 1.5),
            ("", None, 0),
            (f"Echanges detectes : {len(self.rallies)}  (dont {len(longs)} avec retour)", (255, 255, 255), 1.0),
        ]
        if longs:
            lignes.append((f"Echange moyen : {np.mean([r['frappes'] for r in longs]):.1f} frappes | "
                           f"le plus long : {max(r['frappes'] for r in longs)} frappes "
                           f"({max(r['duree_s'] for r in longs):.1f} s)", (255, 255, 255), 1.0))
        lignes += [
            (f"Rythme : {rh:.1f} s entre deux frappes" + (f" ({60 / rh:.0f}/min)" if rh > 0 else ""), (255, 255, 255), 1.0),
            (f"Possession : joueur loin {pf:.0f}%  |  joueur proche {pn:.0f}%", (120, 255, 120), 1.2),
            (f"Frappes : loin {self.hits_by['far']}  |  proche {self.hits_by['near']}", (255, 255, 255), 1.0),
            (f"Services sans retour : loin {self.faults['far']}  |  proche {self.faults['near']}", (80, 120, 255), 1.0),
            *self._var_lines(),
            (f"Vitesse max joueurs : loin {vmax_far * 3.6:.1f} km/h  |  proche {vmax_near * 3.6:.1f} km/h", (255, 255, 255), 1.0),
            (f"Vitesse max balle : ~{ball_max * 3.6:.0f} km/h (estimation)", (0, 255, 255), 1.0),
            ("", None, 0),
            ("Estimations : elles dependent de la detection de la balle.", (170, 170, 170), 0.7),
        ]
        sc = w / 1280
        y = int(h * 0.14)
        for txt, col, k in lignes:
            if txt:
                cv2.putText(img, txt, (int(w * 0.06), y), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.4, 0.8 * k * sc), col, max(1, int(2 * sc)), cv2.LINE_AA)
            y += int(h * 0.075)
        return img

    def _var_lines(self):
        v = getattr(self, "var", None)
        if v is None:
            return []
        return [(f"VAR : {v.counts['IN']} balles IN  |  {v.counts['OUT']} balles OUT  (dont {v.counts['limite']} trop justes)", (90, 255, 140), 1.1),
                (f"Fautes (balle OUT) : loin {v.out_by['far']}  |  proche {v.out_by['near']}", (80, 120, 255), 1.0)]

    def finish(self):
        self.end()

    def print_summary(self):
        longs = [r for r in self.rallies if r["frappes"] >= 2]
        pf, pn = self.possession()
        print("\n=== Possession et rapidite du jeu ===")
        print(f"Echanges detectes : {len(self.rallies)} (dont {len(longs)} avec au moins 2 frappes)")
        if longs:
            print(f"Longueur moyenne  : {np.mean([r['frappes'] for r in longs]):.1f} frappes | "
                  f"plus long : {max(r['frappes'] for r in longs)} frappes "
                  f"({max(r['duree_s'] for r in longs):.1f} s)")
        print(f"Rythme            : {self.rhythm():.2f} s entre deux frappes "
              f"({60 / self.rhythm():.0f} frappes/min)" if self.rhythm() > 0 else "Rythme            : pas assez de frappes")
        print(f"Possession        : loin {pf:.0f}% | proche {pn:.0f}% (temps de balle dans chaque camp)")
        print(f"Frappes detectees : loin {self.hits_by['far']} | proche {self.hits_by['near']}")
        if self.speeds:
            print(f"Vitesse moyenne de la balle pendant les echanges : ~{np.mean(self.speeds) * 3.6:.0f} km/h (estimation)")
        print(f"Balle suivie sur {100.0 * self.n_ball / max(1, self.n_frames):.0f}% des images (objectif : plus de 70 %)")
        print("(estimations : elles dependent du suivi de la balle)")

    def save_csv(self, path):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["debut_s", "duree_s", "frappes", "service", "enchainement"])
            wr.writeheader()
            wr.writerows(self.rallies)
        print(f"Echanges enregistres : {path}  (enchainement : L = joueur loin, P = joueur proche)")


# ---------------------------------------------------------------
# FONCTION 6e : VAR - detecter le REBOND de la balle et juger IN / OUT
#   Rebond = la vitesse verticale de la balle (a l'ecran) saute d'un coup vers le haut,
#            alors que sa direction reste la meme (un coup de raquette, lui, la retourne).
#   Point d'impact = croisement de la trajectoire avant et apres (la balle est alors AU SOL :
#            l'homographie du court donne sa vraie position en metres).
#   Jugement : 1er rebond apres chaque frappe. Jeu normal : dans le simple du camp adverse.
#            Service : dans la zone de service (profondeur) du camp adverse.
#   "limite" = a moins de 15 cm (proche) / 30 cm (loin) d'une ligne : la camera ne permet pas d'etre certain.
# ---------------------------------------------------------------
class VarJudge:
    def __init__(self, fps, width, H):
        self.fps, self.w, self.H = fps, width, H
        self.track = deque(maxlen=6)
        self.run = 0                               # images de suite ou la balle est vue
        self.events = []
        self.last_f = -10 ** 9
        self.seen_hit = -10 ** 9                   # derniere frappe deja "consommee" par un rebond
        self.counts = {"IN": 0, "OUT": 0, "limite": 0}
        self.out_by = {"far": 0, "near": 0}

    def reset(self):
        self.track.clear()
        self.run = 0

    @staticmethod
    def _cross(p1, d1, p2, d2):
        det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
        if abs(det) < 1e-9:
            return None
        bx, by = p2[0] - p1[0], p2[1] - p1[1]
        t = (bx * (-d2[1]) - by * (-d2[0])) / det
        return (p1[0] + t * d1[0], p1[1] + t * d1[1]), t

    def _vscale(self, px):
        """Pixels par metre VERTICAL (hauteur de la balle) autour du point px.
        Deduit de l'homographie : le sol est raccourci en profondeur d'un facteur sin(angle de vue),
        donc un objet debout est a l'echelle  largeur * cos(angle de vue) = largeur * sqrt(1 - (profondeur/largeur)^2)."""
        wx, wy = to_world(self.H, px)
        Hi = np.linalg.inv(self.H)
        def P(x, y):
            q = Hi @ np.array([x, y, 1.0]); return np.array([q[0] / q[2], q[1] / q[2]])
        q0 = P(wx, wy)
        mx = np.hypot(*(P(wx + 0.5, wy) - q0)) / 0.5            # px / m en largeur
        my = np.hypot(*(P(wx, wy + 0.5) - q0)) / 0.5            # px / m en profondeur
        r = min(my / max(mx, 1e-6), 0.97)
        return mx * np.sqrt(1.0 - r * r)

    def _impact(self, p1, p0, p2, p3):
        """Point d'impact au sol. Modele : a l'ecran, la balle suit une parabole (gravite) en hauteur et
        va en ligne droite sur le sol ; on cherche l'instant ou la trajectoire d'avant croise celle d'apres."""
        (f1, x1, y1), (f0, x0, y0), (f2, x2, y2), (f3, x3, y3) = p1, p0, p2, p3
        mid = ((x1 + x2) / 2, (y1 + y2) / 2)
        try:                                       # echelle verticale locale (pixels par metre) -> gravite a l'ecran
            scale = self._vscale(mid)
        except Exception:
            return mid
        a = 9.81 * scale / (self.fps ** 2)          # px / image^2, vers le bas
        d_in, d_out = f1 - f0, f3 - f2
        v1 = (y1 - y0) / d_in + 0.5 * a * d_in      # vitesse verticale juste avant (a l'image f1)
        v2 = (y3 - y2) / d_out - 0.5 * a * d_out    # vitesse verticale juste apres (a l'image f2)
        den = v1 - v2 + a * (f2 - f1)
        if abs(den) < 1e-6:
            return mid
        num = -(y1 - y2) + v1 * f1 - v2 * f2 + 0.5 * a * (f2 - f1) * (f1 + f2)
        tau = num / den
        if not (f1 - TAU_TOL <= tau <= f2 + TAU_TOL):
            return mid
        vx_in, vx_out = (x1 - x0) / d_in, (x3 - x2) / d_out
        xa, xb = x1 + vx_in * (tau - f1), x2 + vx_out * (tau - f2)
        px = ((xa + xb) / 2, y1 + v1 * (tau - f1) + 0.5 * a * (tau - f1) ** 2)
        if np.hypot(px[0] - mid[0], px[1] - mid[1]) > 0.75 * np.hypot(x2 - x1, y2 - y1) + 6:
            return mid                             # l'impact doit se trouver entre les 2 dernieres positions vues
        return px

    def _impact_world(self, pre, post):
        """Point d'impact au sol, calcule en METRES (plan du court) avec la gravite et la parallaxe.
        La balle en l'air apparait decalee par rapport a son ombre au sol : decalage = hauteur x k (k deduit de l'homographie).
        Avant le rebond : coord(t) = lineaire + 0.5*k*g*(t-tau)^2 ; apres : lineaire - 0.5*k*g*(t-tau)^2.
        On cherche le tau (instant du rebond) et le point (x, y) qui collent le mieux a TOUTES les mesures."""
        pts = pre + post
        f1, f2 = pre[-1][0], post[0][0]
        mid = ((pre[-1][1] + post[0][1]) / 2, (pre[-1][2] + post[0][2]) / 2)
        Hi = np.linalg.inv(self.H)
        g = 9.81
        t = np.array([p[0] for p in pts], float) / self.fps
        is_pre = np.array([True] * len(pre) + [False] * len(post))
        sgn = np.where(is_pre, 1.0, -1.0)
        W = np.array([to_world(self.H, (p[1], p[2])) for p in pts], float)
        px_est = mid
        for _ in range(2):                         # k depend du lieu : on affine une fois
            try:
                vs = self._vscale(px_est)
                w0 = np.array(to_world(self.H, px_est))
                jy = np.array(to_world(self.H, (px_est[0], px_est[1] + 1.0))) - w0     # metres par pixel vertical
                jx = np.array(to_world(self.H, (px_est[0] + 1.0, px_est[1]))) - w0     # metres par pixel horizontal
            except Exception:
                return None
            k = -jy * vs                           # metres de decalage (x, y) par metre de hauteur
            gsc = np.array([np.hypot(jx[0], jy[0]), np.hypot(jx[1], jy[1])])           # m/pixel par axe (poids)
            best = None
            for tau in np.linspace(f1 - 0.5, f2 + 0.5, 61) / self.fps:
                dt = t - tau
                A = np.column_stack([np.ones_like(dt), np.where(is_pre, dt, 0.0), np.where(~is_pre, dt, 0.0)])
                sse, sol = 0.0, []
                for ax in (0, 1):
                    z = W[:, ax] - sgn * 0.5 * k[ax] * g * dt ** 2
                    c, *_ = np.linalg.lstsq(A / gsc[ax], z / gsc[ax], rcond=None)
                    sse += float(np.sum(((A @ c - z) / gsc[ax]) ** 2))
                    sol.append(c[0])
                if best is None or sse < best[0]:
                    best = (sse, sol[0], sol[1])
            q = Hi @ np.array([best[1], best[2], 1.0])
            px_est = (q[0] / q[2], q[1] / q[2])
        seg = np.hypot(post[0][1] - pre[-1][1], post[0][2] - pre[-1][2])
        if np.hypot(px_est[0] - mid[0], px_est[1] - mid[1]) > 1.0 * seg + 8:
            return None
        return px_est

    def _impact_fit(self, pre, post):
        """Meilleure estimation du point d'impact avec TOUS les points (4 avant + 2 apres) :
        pour chaque instant de rebond possible tau, on ajuste 2 droites (sol, en x) et 2 paraboles (en y, gravite connue)
        qui se rejoignent au meme point ; on garde le tau qui colle le mieux aux mesures."""
        pts = pre + post
        f1, f2 = pre[-1][0], post[0][0]
        mid = ((pre[-1][1] + post[0][1]) / 2, (pre[-1][2] + post[0][2]) / 2)
        try:
            scale = self._vscale(mid)
        except Exception:
            return None
        a = 9.81 * scale / (self.fps ** 2)
        t = np.array([p[0] for p in pts], float)
        X = np.array([p[1] for p in pts], float)
        Y = np.array([p[2] for p in pts], float)
        is_pre = np.array([True] * len(pre) + [False] * len(post))
        best = None
        for tau in np.linspace(f1, f2, 41):
            dt = t - tau
            A = np.column_stack([np.ones_like(dt), np.where(is_pre, dt, 0.0), np.where(~is_pre, dt, 0.0)])
            cy, ry, *_ = np.linalg.lstsq(A, Y - 0.5 * a * dt ** 2, rcond=None)
            cx, rx, *_ = np.linalg.lstsq(A, X, rcond=None)
            sse = float(np.sum((A @ cy - (Y - 0.5 * a * dt ** 2)) ** 2) + np.sum((A @ cx - X) ** 2))
            if best is None or sse < best[0]:
                best = (sse, cx[0], cy[0])
        px = (best[1], best[2])
        seg = np.hypot(post[0][1] - pre[-1][1], post[0][2] - pre[-1][2])
        if np.hypot(px[0] - mid[0], px[1] - mid[1]) > 0.75 * seg + 6:
            return None
        return px

    def update(self, f, ball_px, rally, person_boxes=()):
        if ball_px is None:
            return None
        if self.track and f - self.track[-1][0] > BOUNCE_MAX_GAP:
            self.track.clear()
            self.run = 0
        self.track.append((f, ball_px[0], ball_px[1]))
        self.run += 1
        if len(self.track) < 4 or self.run < BOUNCE_MIN_RUN or f - self.last_f < 0.4 * self.fps:
            return None
        if VAR_NEEDS_RALLY and not rally.active:
            return None
        trk = list(self.track)
        (f0, x0, y0), (f1, x1, y1), (f2, x2, y2), (f3, x3, y3) = trk[-4:]
        if rally.last_hit >= f1 - 1 and rally.last_hit <= f2 + 1:
            return None                            # c'est un coup de raquette, pas un rebond
        vin = ((x1 - x0) / (f1 - f0), (y1 - y0) / (f1 - f0))
        vout = ((x3 - x2) / (f3 - f2), (y3 - y2) / (f3 - f2))
        nin, nout = np.hypot(*vin), np.hypot(*vout)
        if min(nin, nout) < BOUNCE_MIN_SPEED * self.w:
            return None
        if (vin[0] * vout[0] + vin[1] * vout[1]) / (nin * nout) < BOUNCE_COS:
            return None
        if not (0.25 <= nout / nin <= 4.0):
            return None                            # vitesse qui change trop : ce n'est pas la meme balle
        if vout[1] - vin[1] > -BOUNCE_KICK * self.w:
            return None                            # pas de saut vers le haut : pas de rebond
        hitter = rally.last_hitter if rally.active else None
        if hitter is not None and rally.last_hit <= self.seen_hit:
            return None                            # le 1er rebond apres cette frappe est deja juge
        pt4 = self._impact((f1, x1, y1), (f0, x0, y0), (f2, x2, y2), (f3, x3, y3))
        pre_n = 4 if len(trk) >= 6 else (3 if len(trk) == 5 else 2)
        ptw = self._impact_world(trk[-(pre_n + 2):-2], trk[-2:]) if VAR_IMPACT_MODE in ("world", "mean") else None
        ptf = self._impact_fit(trk[:-2], trk[-2:]) if (len(trk) >= 5 and self.run >= 5 and VAR_IMPACT_MODE == "fit") else None
        if VAR_IMPACT_MODE == "world" and ptw is not None:
            pt = ptw
        elif VAR_IMPACT_MODE == "fit" and ptf is not None:
            pt = ptf
        elif VAR_IMPACT_MODE == "mean" and ptw is not None:
            pt = ((pt4[0] + ptw[0]) / 2, (pt4[1] + ptw[1]) / 2)
        else:
            pt = pt4
        for bx1, by1, bx2, by2 in person_boxes:    # impact colle a un joueur : raquette / pieds, pas un rebond
            m = PLAYER_REACH * (by2 - by1)
            if bx1 - m <= pt[0] <= bx2 + m and by1 - m <= pt[1] <= by2 + m:
                return None
        wx, wy = to_world(self.H, pt)
        return self._judge(f, pt, (wx, wy), hitter, len(rally.hitters) if rally.active else 0, rally)

    def _judge(self, f, px, world, hitter, n_hits, rally):
        wx, wy = world
        net = COURT_L / 2
        s_in = 1.37
        if hitter is not None:
            if (wy < net) == (hitter == "far"):
                return None                        # rebond dans le camp du frappeur : parasite
            kind = "service" if n_hits == 1 else "jeu"
        else:
            if not (-3 < wy < COURT_L + 3):
                return None
            kind = "jeu?"
        far_half = wy < net
        if kind == "service":
            depth = 6.40 - abs(wy - net)           # distance a la ligne de service (>0 = dans la zone)
        else:
            depth = wy if far_half else COURT_L - wy      # distance a la ligne de fond
        terms = {"x": min(wx - s_in, (COURT_W - s_in) - wx), "y": depth}
        axis = min(terms, key=terms.get)           # la ligne la plus proche : laterale (x) ou de fond / service (y)
        d = terms[axis]                            # >0 dedans, <0 dehors (metres)
        if d < -VAR_MAX_OUT:
            return None                            # a plus de 2.5 m des lignes : ce n'est pas un vrai rebond (fausse balle)
        # incertitude : combien de metres valent SIGMA_PX pixels ici (grande au fond du court, petite pres de la camera)
        try:
            w0 = np.array(to_world(self.H, px))
            jx = np.array(to_world(self.H, (px[0] + 2.0, px[1]))) - w0
            jy = np.array(to_world(self.H, (px[0], px[1] + 2.0))) - w0
            k = 0 if axis == "x" else 1
            extra = VAR_NEAR_EXTRA * float(np.clip((wy - 16.0) / 6.0, 0.0, 1.0))        # proche de la camera : pas de balle tres grands
            unc = float(np.sqrt((SIGMA_PX / 2.0 * np.hypot(jx[k], jy[k])) ** 2 + VAR_MODEL_ERR ** 2 + extra ** 2))
        except Exception:
            unc = 0.30
        verdict = "IN" if d >= -BALL_RADIUS else "OUT"
        limite = abs(d) < unc
        ev = {"frame": f, "t": f / self.fps, "px": px, "world": (wx, wy), "hitter": hitter, "kind": kind,
              "verdict": verdict, "d": d, "limite": limite, "far_half": far_half, "unc": unc}
        self.events.append(ev)
        self.last_f = f
        self.seen_hit = rally.last_hit
        self.counts[verdict] += 1
        if limite:
            self.counts["limite"] += 1
        if verdict == "OUT" and hitter is not None:
            self.out_by[hitter] += 1
        return ev

    def judge_click(self, f, px, service=False):
        """Decision MANUELLE : l'utilisateur a clique l'endroit ou la balle touche le sol (video arretee par VAR).
        Meme calcul que la decision automatique (lignes de touche, fond ou service, incertitude), sans detecter la balle."""
        wx, wy = to_world(self.H, px)
        if not (-4.0 < wx < COURT_W + 4.0 and -6.0 < wy < COURT_L + 6.0):
            return None                            # clic loin du terrain
        net = COURT_L / 2
        far_half = wy < net
        s_in = 1.37
        depth = (6.40 - abs(wy - net)) if service else (wy if far_half else COURT_L - wy)
        terms = {"x": min(wx - s_in, (COURT_W - s_in) - wx), "y": depth}
        axis = min(terms, key=terms.get)
        d = terms[axis]
        try:
            w0 = np.array(to_world(self.H, px))
            jx = np.array(to_world(self.H, (px[0] + 2.0, px[1]))) - w0
            jy = np.array(to_world(self.H, (px[0], px[1] + 2.0))) - w0
            k = 0 if axis == "x" else 1
            unc = float(np.sqrt((CLICK_SIGMA_PX / 2.0 * np.hypot(jx[k], jy[k])) ** 2 + VAR_MODEL_ERR ** 2))
        except Exception:
            unc = 0.30
        verdict = "IN" if d >= -BALL_RADIUS else "OUT"
        ev = {"frame": f, "t": f / self.fps, "px": px, "world": (wx, wy), "hitter": None,
              "kind": "manuel", "verdict": verdict, "d": d, "limite": abs(d) < unc, "far_half": far_half, "unc": unc}
        self.events.append(ev)
        self.counts[verdict] += 1
        if ev["limite"]:
            self.counts["limite"] += 1
        return ev

    def last(self):
        return self.events[-1] if self.events else None

    def save_csv(self, path):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["temps_s", "frappeur", "type", "x_m", "y_m", "decision", "distance_ligne_m", "incertitude_m", "limite"])
            for e in self.events:
                wr.writerow([round(e["t"], 1), RallyTracker.NOM.get(e["hitter"], "?"), e["kind"],
                             round(e["world"][0], 2), round(e["world"][1], 2), e["verdict"],
                             round(e["d"], 2), round(e["unc"], 2), "oui" if e["limite"] else "non"])
        print(f"Decisions VAR enregistrees : {path}")


def draw_var(frame, ev, age_s, text_scale):
    """Marque du point d'impact + bandeau de decision (affiches VAR_SHOW_SECONDS secondes)."""
    fh, fw = frame.shape[:2]
    ok = ev["verdict"] == "IN"
    col = (60, 220, 90) if ok else (255, 70, 60)                          # RGB
    bgr = (col[2], col[1], col[0])
    x, y = int(ev["px"][0]), int(ev["px"][1])
    u = max(0.6, fw / 1920)
    pulse = 1.0 + 0.25 * np.sin(age_s * 8)
    ax, ay = int(26 * u * pulse), int(10 * u * pulse)                     # ellipse = cercle vu en perspective
    cv2.ellipse(frame, (x, y), (ax, ay), 0, 0, 360, (0, 0, 0), max(3, int(5 * u)), cv2.LINE_AA)
    cv2.ellipse(frame, (x, y), (ax, ay), 0, 0, 360, bgr, max(2, int(3 * u)), cv2.LINE_AA)
    cv2.circle(frame, (x, y), max(2, int(3 * u)), bgr, -1, cv2.LINE_AA)
    txt1 = "VAR  •  " + (VAR_WORD_IN if ok else "OUT")
    dist = abs(ev["d"]) * 100
    txt2 = (f"{'dedans' if ok else 'dehors'} de {dist:.0f} cm (± {ev['unc'] * 100:.0f} cm)" + ("  •  trop juste pour trancher" if ev["limite"] else "")
            + (f"  •  frappe de {RallyTracker.NOM[ev['hitter']]}" if ev["hitter"] else ""))
    if Image is None:
        cv2.putText(frame, f"{txt1} {txt2}", (int(fw * 0.3), int(fh * 0.12)), cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale * 1.2, bgr, 3, cv2.LINE_AA)
        return
    f1, f2 = _font(60 * u, True), _font(26 * u, False)
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    pw = int(max(tmp.textlength(txt1, font=f1), tmp.textlength(txt2, font=f2)) + 70 * u)
    ph = int(125 * u)
    x0, y0 = (fw - pw) // 2, int(fh * 0.94) - ph          # en bas au centre : ne cache ni le panneau ni la carte
    roi = frame[y0:y0 + ph, x0:x0 + pw]
    base = _to_pil(roi)
    base.alpha_composite(_card((pw, ph), int(18 * u), fill=(10, 14, 22, 225), outline=(*col, 255)))
    dr = ImageDraw.Draw(base)
    dr.rounded_rectangle((0, 0, pw - 1, int(9 * u)), 4, fill=(*col, 255))
    dr.text((pw / 2, int(54 * u)), txt1, font=f1, fill=col, anchor="mm")
    dr.text((pw / 2, int(100 * u)), txt2, font=f2, fill=(235, 235, 235), anchor="mm")
    frame[y0:y0 + ph, x0:x0 + pw] = _from_pil(base)


def save_var_image(frame, ev, n):
    os.makedirs(VAR_DIR, exist_ok=True)
    fh, fw = frame.shape[:2]
    x, y = int(ev["px"][0]), int(ev["px"][1])
    rx, ry = int(0.14 * fw), int(0.10 * fw)
    crop = frame[max(0, y - ry):min(fh, y + ry), max(0, x - rx):min(fw, x + rx)]
    crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    name = os.path.join(VAR_DIR, f"var_{n:02d}_{ev['verdict']}{'_limite' if ev['limite'] else ''}.png")
    cv2.imwrite(name, crop)
    return name


# ---------------------------------------------------------------
# FONCTION 7 : programme principal
# ---------------------------------------------------------------
def main(video_path):
    # --- Ouverture de la video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Impossible d'ouvrir {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    eff_fps = fps / STRIDE                         # cadence reelle des images traitees
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    text_scale = max(0.5, w / 1920 * 0.9)          # texte proportionnel a la video

    # --- Calibration du court ---
    ok, first = cap.read()
    if CALIB_POINTS:                               # calibration precise (6 points)
        H, corners, cerr = calib_from_points(CALIB_POINTS)
        print(f"Calibration : 6 points, erreur moyenne {cerr:.1f} px")
    elif CORNERS:                                  # ancienne methode : 4 coins
        corners = np.float32(CORNERS)
        H = cv2.getPerspectiveTransform(corners, np.float32([(0, 0), (COURT_W, 0), (COURT_W, COURT_L), (0, COURT_L)]))
    elif CALIB_MODE == "6":
        H, corners, cerr = calib_from_points(pick_calibration(first))
    else:                                          # par defaut : on clique les 4 coins, comme avant
        corners = pick_corners(first)
        H = cv2.getPerspectiveTransform(corners, np.float32([(0, 0), (COURT_W, 0), (COURT_W, COURT_L), (0, COURT_L)]))
    cx0, cy0, cx1, cy1 = court_crop(np.clip(corners, [-w, -h], [2 * w, 2 * h]), w, h) if CROP_TO_COURT else (0, 0, w, h)
    court_ratio = make_court_checker(first, corners)
    height_ratio = make_height_checker(H)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- Sortie video et modele ---
    out = cv2.VideoWriter("output_tennis.mp4", cv2.VideoWriter_fourcc(*"mp4v"), eff_fps, (w, h))
    model = YOLO(MODEL_NAME)
    ball_model = YOLO(BALL_MODEL) if BALL_MODEL else None
    if ball_model:
        print(f"Modele balle : {BALL_MODEL} (classe {BALL_MODEL_CLASS}, imgsz={BALL_IMG_SIZE})")
    print(f"Modele {MODEL_NAME} | device={DEVICE} | imgsz={IMG_SIZE} | stride={STRIDE} | crop={cx0, cy0, cx1, cy1}")

    # --- Variables joueurs ---
    history = {"far": deque(maxlen=WINDOW + 1), "near": deque(maxlen=WINDOW + 1)}
    max_speed = {"far": 0.0, "near": 0.0}
    cur_speed = {"far": 0.0, "near": 0.0}
    last_pos = {"far": None, "near": None}         # derniere position (metres) de chaque joueur suivi
    last_seen = {"far": -10**9, "near": -10**9}    # derniere image ou il a ete vu
    last_box = {"far": None, "near": None}        # derniere boite vue (pour garder l'etiquette quand la detection saute)

    # --- Variables balle ---
    ball_last = None
    ball_vel = (0.0, 0.0)
    ball_missed = 0
    ball_trail = deque(maxlen=TRAIL_LEN)
    ball_hist = deque(maxlen=BALL_WINDOW + 1)      # (numero image, x, y en metres)
    ball_speed = 0.0
    ball_max = 0.0
    max_jump = min(BALL_JUMP_FRAC * w * STRIDE, 0.22 * w)   # le saut grandit si on saute des images, mais plafonne
    static_cells = {}                              # compte les candidats immobiles (objets fixes)
    prev_gray = None
    ball_world_trail = deque(maxlen=TRAIL_LEN)

    judge_ref = [None]
    viewer = Viewer(w, h, eff_fps, SHOW_WINDOW)
    viewer.manual_cb = lambda f_, px_: judge_ref[0].judge_click(f_, px_) if judge_ref[0] else None
    hud_top = Viewer.reserved_top(w)               # place pour le rectangle VAR / DECISION au-dessus des stats
    court_poly = corners.astype(np.int32).reshape(-1, 1, 2)
    rally = RallyTracker(eff_fps, w)
    judge = VarJudge(eff_fps, w, H)
    judge_ref[0] = judge
    rally.var = judge
    var_show = None                                # (decision, image ou elle a ete prise)
    frame_idx = 0
    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for _ in range(STRIDE - 1):                # saute des images sans les decoder
            cap.grab()

        # ---- Vue large du court ? Sinon : on met l'analyse en pause ----
        ratio = court_ratio(frame)
        if ratio < MAIN_VIEW_MIN:
            for side in history:
                history[side].clear()
                cur_speed[side] = 0.0
            last_pos["far"] = last_pos["near"] = None
            last_box["far"] = last_box["near"] = None
            rally.interrupt()
            judge.reset()
            ball_last = None
            ball_vel = (0.0, 0.0)
            ball_trail.clear()
            ball_hist.clear()
            ball_world_trail.clear()
            prev_gray = None
            ball_speed = 0.0
            cv2.putText(frame, "Hors vue du court : analyse en pause", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * text_scale + 0.3, (0, 0, 255), 2, cv2.LINE_AA)
            out.write(frame)
            viewer.push(frame)
            if not viewer.tick():
                break
            frame_idx += 1
            continue

        if SHOW_COURT_OUTLINE:                     # terrain de detection : seuls les 2 joueurs dedans sont suivis
            draw_court_lines(frame, H, (255, 220, 0), max(1, int(text_scale * 1.5)))

        # Une seule passe YOLO : personnes (0) et balle (32)
        player_cands = []                          # toutes les personnes dans la zone du court
        ball_cands = []
        crop_img = frame[cy0:cy1, cx0:cx1]
        res = model.predict(crop_img, classes=([0] if ball_model else [0, BALL_CLASS]),
                            conf=BALL_CONF, imgsz=IMG_SIZE, device=DEVICE,
                            verbose=False)[0]
        if ball_model:                             # 2e passe : modele specialise "balle", a plus haute resolution
            resb = ball_model.predict(crop_img, conf=BALL_CONF, imgsz=BALL_IMG_SIZE, device=DEVICE, verbose=False)[0]
            for bb in resb.boxes:
                if int(bb.cls[0]) != BALL_MODEL_CLASS:
                    continue
                bx1, by1, bx2, by2 = bb.xyxy[0].tolist()
                bcx, bcy = (bx1 + bx2) / 2 + cx0, (by1 + by2) / 2 + cy0
                bwx, bwy = to_world(H, (bcx, bcy))
                if in_ball_zone(bwx, bwy):
                    ball_cands.append((bcx, bcy, float(bb.conf[0])))

        if COLOR_BALL:
            extra, prev_gray = color_ball_candidates(frame[cy0:cy1, cx0:cx1], prev_gray, cx0, cy0, w)
            for c in extra:
                wx_, wy_ = to_world(H, (c[0], c[1]))
                if in_ball_zone(wx_, wy_):
                    ball_cands.append(c)
        person_boxes = []                          # toutes les personnes vues (pour rejeter les fausses balles)

        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, x2, y1, y2 = x1 + cx0, x2 + cx0, y1 + cy0, y2 + cy0   # retour aux coordonnees de l'image entiere
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls == BALL_CLASS:                  # --- candidat balle ---
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                wx, wy = to_world(H, (cx, cy))
                if in_ball_zone(wx, wy):
                    ball_cands.append((cx, cy, conf))
                continue

            if conf < PERSON_CONF_TRACK:
                continue
            person_boxes.append((x1, y1, x2, y2))
            foot = ((x1 + x2) / 2, y2)
            wx, wy = to_world(H, foot)
            # zone stricte autour du court : exclut le public, les juges de ligne lointains
            my_ = PLAYER_MARGIN_Y_FAR if wy < COURT_L / 2 else PLAYER_MARGIN_Y_NEAR
            if not (-PLAYER_MARGIN_X < wx < COURT_W + PLAYER_MARGIN_X and -my_ < wy < COURT_L + my_):
                continue
            if y1 > 3 and not (PLAYER_HEIGHT_RANGE[0] <= height_ratio((wx, wy), (x1, y1, x2, y2)) <= PLAYER_HEIGHT_RANGE[1]):   # (joueur coupe en haut de l'image : on ne juge pas sa taille)
                continue                           # trop petit (ramasseur de balles) ou trop grand : pas un joueur
            player_cands.append(((x2 - x1) * (y2 - y1), foot, (wx, wy), (x1, y1, x2, y2), conf))

        # --- Joueurs : on garde ceux deja suivis, on ignore les nouveaux venus ---
        lost_max = LOST_SECONDS * eff_fps
        locked_before = {s: last_pos[s] is not None and frame_idx - last_seen[s] <= lost_max
                         for s in ("far", "near")}
        best = assign_players(player_cands, last_pos, last_seen, frame_idx, eff_fps)
        for side in ("far", "near"):
            if best[side] is not None:
                if not locked_before[side] or frame_idx - last_seen[side] > COAST_SECONDS * eff_fps:
                    history[side].clear()          # nouveau joueur ou trou trop long : on repart a zero
                    cur_speed[side] = 0.0
                last_pos[side] = best[side][2]
                last_seen[side] = frame_idx
                last_box[side] = best[side][3]
            elif last_pos[side] is not None and not locked_before[side]:
                last_pos[side] = None

        # --- Balle : on rejette les candidats poses sur une personne (chaussure, raquette, short) ---
        def sur_une_personne(c):
            return any(bx1 - 4 <= c[0] <= bx2 + 4 and by1 - 4 <= c[1] <= by2 + 4
                       for bx1, by1, bx2, by2 in person_boxes)
        for side_ in ("far", "near"):              # joueur non detecte cette image : sa derniere boite sert quand meme de filtre
            if best[side_] is None and last_box[side_] is not None and frame_idx - last_seen[side_] <= COAST_SECONDS * eff_fps:
                person_boxes.append(last_box[side_])
        ball_cands = [c for c in ball_cands if not sur_une_personne(c)]

        def pres_dun_joueur(c):                    # zone large : la raquette depasse de la boite du joueur
            for bx1, by1, bx2, by2 in person_boxes:
                m = PLAYER_REACH * (by2 - by1)
                if bx1 - m <= c[0] <= bx2 + m and by1 - m <= c[1] <= by2 + m:
                    return True
            return False
        ball_cands = [c for c in ball_cands if not (len(c) > 3 and pres_dun_joueur(c))]

        # --- Balle : on rejette les objets fixes (balles au sol, seau a balles...) ---
        gardes = []
        for c in ball_cands:
            key = (int(c[0] // 12), int(c[1] // 12))
            static_cells[key] = static_cells.get(key, 0) + 1
            if static_cells[key] < STATIC_MAX:
                gardes.append(c)
        ball_cands = gardes

        # --- Joueurs : positions et vitesses (etiquette continue, meme si la detection saute 1 s) ---
        positions = {}
        for side in ("far", "near"):
            coast = False
            if best[side] is not None:
                _, foot, world, box = best[side][:4]
                update_speed(history, cur_speed, max_speed, side, frame_idx, world, eff_fps)
            elif last_box[side] is not None and frame_idx - last_seen[side] <= COAST_SECONDS * eff_fps:
                box, world, coast = last_box[side], last_pos[side], True
                if world is None:
                    positions[side] = None
                    continue
            else:
                positions[side] = None
                continue
            positions[side] = world
            ready = cur_speed[side] > 0 or len(history[side]) > WINDOW
            color = (0, 200, 255) if side == "far" else (255, 120, 0)
            draw_player_marker(frame, box, color, cur_speed[side] * 3.6 if ready else None, text_scale)

        # --- Balle : choix, trous comblés, vitesse ---
        ball_world = None
        chosen = select_ball(ball_cands, ball_last, max_jump)

        if chosen is not None:
            if ball_last is not None:
                ball_vel = (chosen[0] - ball_last[0], chosen[1] - ball_last[1])
            ball_last = chosen
            ball_missed = 0
            ball_trail.append(chosen)
            ball_world = to_world(H, chosen)
            ball_hist.append((frame_idx, ball_world[0], ball_world[1]))

            if len(ball_hist) >= 3:
                f0, x0, y0 = ball_hist[0]
                f1, x1_, y1_ = ball_hist[-1]
                if f1 > f0:
                    v = np.hypot(x1_ - x0, y1_ - y0) / ((f1 - f0) / eff_fps)
                    if v <= BALL_MAX_SPEED:
                        ball_speed = v
                        ball_max = max(ball_max, v)

        elif ball_last is not None and ball_missed < BALL_MAX_MISSED:
            ball_last = (ball_last[0] + ball_vel[0], ball_last[1] + ball_vel[1])
            ball_missed += 1
        else:
            ball_last = None
            ball_vel = (0.0, 0.0)
            ball_trail.clear()
            ball_hist.clear()
            ball_speed = 0.0

        draw_ball(frame, ball_trail, show_head=(ball_missed == 0 and ball_last is not None))

        boxes = {s_: (best[s_][3] if best[s_] is not None else None) for s_ in ("far", "near")}
        rally.update(frame_idx, chosen, ball_world, ball_speed, boxes)
        info = rally.live_data()
        info.update(vmax_far=max_speed["far"] * 3.6, vmax_near=max_speed["near"] * 3.6,
                    ball_kmh=ball_speed * 3.6, ball_max=ball_max * 3.6)
        ev = judge.update(frame_idx, chosen, rally, person_boxes) if VAR_ENABLED else None
        if ev is not None:
            var_show = ev
            print(f"VAR  t={ev['t']:.1f}s  {ev['verdict']}  ({ev['world'][0]:.2f} m, {ev['world'][1]:.2f} m)  "
                  f"{abs(ev['d']) * 100:.0f} cm (+/- {ev['unc'] * 100:.0f}) {'dedans' if ev['verdict'] == 'IN' else 'dehors'}"
                  f"{'  [limite]' if ev['limite'] else ''}")
        last = judge.last()
        info.update(var_in=judge.counts["IN"], var_out=judge.counts["OUT"],
                    var_last=(last["verdict"] if last else "—"),
                    out_far=judge.out_by["far"], out_near=judge.out_by["near"])
        pw_hud = draw_hud(frame, info, hud_top)
        if pw_hud:
            viewer.hud_w = pw_hud + 10
        if var_show is not None:
            age = (frame_idx - var_show["frame"]) / eff_fps
            if age <= VAR_SHOW_SECONDS:
                draw_var(frame, var_show, age, text_scale)
        ball_world_trail.append(ball_world)
        draw_minimap(frame, positions, ball_world, list(ball_world_trail), judge.events[-6:])
        var_path = None
        if ev is not None:
            var_path = save_var_image(frame, ev, len(judge.events))
            if VAR_AUTO_PAUSE and ev["verdict"] == "OUT":
                viewer.paused = True

        out.write(frame)
        viewer.push(frame, frame_idx)
        if ev is not None:
            viewer.add_var_event(ev, var_path)     # le bouton VAR de la fenetre pointe vers cette decision
        if not viewer.tick():                      # pause / retour / zoom geres ici ; False = quitter
            break
        frame_idx += 1

        # --- Progression ---
        if frame_idx % 50 == 0:
            el = time.time() - t_start
            msg = f"{frame_idx} images traitees | {frame_idx / el:.1f} img/s"
            if total:
                restant = (total / STRIDE - frame_idx) / (frame_idx / el)
                msg += f" | reste ~{restant / 60:.1f} min"
            print(msg)

    # --- Fin ---
    cap.release()
    rally.finish()
    out.release()
    print("\n=== Resume ===")
    print(f"Joueur loin  : vitesse max {max_speed['far'] * 3.6:.1f} km/h")
    print(f"Joueur proche: vitesse max {max_speed['near'] * 3.6:.1f} km/h")
    print(f"Balle        : vitesse max ~{ball_max * 3.6:.0f} km/h (estimation)")
    print("Video annotee : output_tennis.mp4")
    rally.print_summary()
    rally.save_csv(STATS_CSV)
    if VAR_ENABLED:
        judge.save_csv(VAR_CSV)
    viewer.review()                                # la fenetre reste ouverte pour revoir / zoomer (q pour fermer)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage : python tennis_analyzer.py match.mp4")
    main(sys.argv[1])