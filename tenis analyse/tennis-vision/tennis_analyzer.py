"""
Tennis Analyzer - Etape 1 : joueurs + vitesse (test sur video)

Usage :
    pip install ultralytics opencv-python numpy
    python tennis_analyzer.py ..\match.mp4

Au demarrage, clique les 4 coins du court (terrain double) sur la 1ere image :
    1) coin lointain gauche   2) coin lointain droit
    3) coin proche droit      4) coin proche gauche
Appuie sur 'q' pendant la lecture pour arreter.
Sortie : output_tennis.mp4 + resume des vitesses max dans la console.
"""
import sys
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------
# REGLAGES
# ---------------------------------------------------------------
COURT_W, COURT_L = 10.97, 23.77   # dimensions reelles du court de double (metres)
MAX_HUMAN_SPEED = 10.0            # m/s : au-dela, c'est une erreur de detection
WINDOW = 5                        # nombre d'images pour calculer une vitesse stable
MINIMAP_SCALE = 14                # mini-carte : 1 metre = 14 pixels
TEXT_SIZE = 0.45                  # taille du texte des vitesses (plus petit = plus discret)


# ---------------------------------------------------------------
# FONCTION 1 : choisir les 4 coins du court avec la souris
# Entree : la 1ere image de la video
# Sortie : 4 points (x, y) en pixels, dans l'echelle de la vraie video
# ---------------------------------------------------------------
def pick_corners(frame):
    w = frame.shape[1]            # largeur de l'image en pixels
    s = min(1.0, 1280 / w)        # facteur de reduction pour que l'image tienne sur l'ecran
    pts = []                      # liste qui recevra les 4 clics

    # Fonction appelee automatiquement a chaque evenement de souris
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:   # clic gauche, max 4 points
            pts.append((x, y))

    win = "Clique les 4 coins du court (ESC pour annuler)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)   # branche la souris sur la fenetre
    labels = ["loin-gauche", "loin-droit", "proche-droit", "proche-gauche"]

    while True:
        img = cv2.resize(frame, None, fx=s, fy=s)   # image reduite pour l'affichage

        # Dessine un point rouge + un nom sur chaque coin deja clique
        for i, p in enumerate(pts):
            cv2.circle(img, p, 6, (0, 0, 255), -1)
            cv2.putText(img, labels[i], (p[0] + 8, p[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

        # Indique quel coin cliquer maintenant
        if len(pts) < 4:
            cv2.putText(img, f"Coin : {labels[len(pts)]}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF    # attend 20 ms une touche du clavier
        if len(pts) == 4:             # les 4 coins sont choisis -> on sort
            break
        if k == 27:                   # touche ESC -> on annule tout
            cv2.destroyAllWindows()
            sys.exit("Annule.")

    cv2.destroyWindow(win)
    # Remet les clics a l'echelle de la vraie video (on avait reduit l'image de s)
    return np.float32([(x / s, y / s) for x, y in pts])


# ---------------------------------------------------------------
# FONCTION 2 : convertir un point pixel en position reelle (metres)
# Entree : H = matrice d'homographie, pt = (x, y) en pixels
# Sortie : (x, y) en metres sur le court
# ---------------------------------------------------------------
def to_world(H, pt):
    p = H @ np.array([pt[0], pt[1], 1.0])   # multiplication matricielle (coordonnees homogenes)
    return p[0] / p[2], p[1] / p[2]         # on divise par p[2] pour corriger la perspective


# ---------------------------------------------------------------
# FONCTION 3 : dessiner la mini-carte du court vue de dessus
# Entree : l'image video, et la position des deux joueurs en metres
# Effet : colle la mini-carte en bas a droite de l'image
# ---------------------------------------------------------------
def draw_minimap(frame, positions):
    mw, mh = int(COURT_W * MINIMAP_SCALE), int(COURT_L * MINIMAP_SCALE)  # taille en pixels
    mm = np.full((mh, mw, 3), (60, 120, 60), np.uint8)                  # fond vert
    cv2.rectangle(mm, (0, 0), (mw - 1, mh - 1), (255, 255, 255), 2)     # contour du court
    cv2.line(mm, (0, mh // 2), (mw, mh // 2), (255, 255, 255), 2)       # ligne du filet
    colors = {"far": (0, 200, 255), "near": (255, 120, 0)}              # une couleur par joueur

    for side, pos in positions.items():
        if pos is not None:   # on dessine le joueur seulement s'il est detecte
            x = int(np.clip(pos[0], 0, COURT_W) * MINIMAP_SCALE)   # clip = reste dans le court
            y = int(np.clip(pos[1], 0, COURT_L) * MINIMAP_SCALE)
            cv2.circle(mm, (x, y), 6, colors[side], -1)

    # Colle la mini-carte dans le coin bas-droit de la video
    fh, fw = frame.shape[:2]
    frame[fh - mh - 10: fh - 10, fw - mw - 10: fw - 10] = mm


# ---------------------------------------------------------------
# FONCTION 4 : dessiner un petit rectangle au-dessus de la tete avec la vitesse
# Entree : image, boite du joueur (x1, y1, x2, y2), couleur, vitesse (km/h)
# ---------------------------------------------------------------
def draw_player_marker(frame, box, color, speed_kmh):
    x1, y1, x2, y2 = [int(v) for v in box]
    cx = (x1 + x2) // 2                                   # milieu horizontal du joueur
    text = f"{speed_kmh:.1f} km/h"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, 1)  # taille du texte
    pad = 4                                               # marge autour du texte

    # Rectangle place juste au-dessus de la tete (haut de la boite - 6 pixels)
    bx1, bx2 = cx - tw // 2 - pad, cx + tw // 2 + pad
    by2 = y1 - 6
    by1 = by2 - th - 2 * pad
    if by1 < 0:                                           # evite de sortir en haut de l'image
        by1 = 0
        by2 = th + 2 * pad

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)   # -1 = rectangle rempli

    # Texte noir sur couleur claire, blanc sur couleur foncee
    text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
    cv2.putText(frame, text, (bx1 + pad, by2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, text_color, 1)


# ---------------------------------------------------------------
# FONCTION 5 : programme principal
# Lit la video image par image, detecte les joueurs, calcule les vitesses
# ---------------------------------------------------------------
def main(video_path):
    # --- Ouverture de la video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Impossible d'ouvrir {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25            # images par seconde (25 si inconnu)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Calibration du court ---
    ok, first = cap.read()                           # 1ere image
    corners = pick_corners(first)                    # tu cliques les 4 coins
    # Position reelle des 4 coins en metres (meme ordre que les clics)
    dst = np.float32([(0, 0), (COURT_W, 0), (COURT_W, COURT_L), (0, COURT_L)])
    H = cv2.getPerspectiveTransform(corners, dst)    # calcule l'homographie pixels -> metres
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)              # revient au debut de la video

    # --- Preparation de la sortie et du modele ---
    out = cv2.VideoWriter("output_tennis.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    model = YOLO("yolov8n.pt")                       # modele de detection (petit et rapide)

    # --- Variables pour les vitesses ---
    history = {"far": deque(maxlen=WINDOW + 1), "near": deque(maxlen=WINDOW + 1)}  # dernieres positions
    max_speed = {"far": 0.0, "near": 0.0}            # vitesse max de chaque joueur (m/s)
    cur_speed = {"far": 0.0, "near": 0.0}            # vitesse actuelle (m/s)
    view_scale = min(1.0, 1280 / w)                  # reduction pour l'affichage a l'ecran

    # --- Boucle : une iteration = une image de la video ---
    while True:
        ok, frame = cap.read()
        if not ok:                                   # fin de la video
            break

        # Detection + suivi des personnes (classes=[0] = "personne" uniquement)
        res = model.track(frame, persist=True, classes=[0], verbose=False)[0]

        # Pour chaque cote du filet, on garde la meilleure detection
        best = {"far": None, "near": None}
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()    # coins de la boite en pixels
            conf = float(box.conf[0])                # confiance de la detection
            foot = ((x1 + x2) / 2, y2)               # point des pieds = milieu du bas de la boite
            wx, wy = to_world(H, foot)               # position des pieds en metres
            if not (-3 < wx < COURT_W + 3 and -5 < wy < COURT_L + 5):
                continue                             # hors du court : spectateur, juge de ligne...
            side = "far" if wy < COURT_L / 2 else "near"   # cote loin ou proche du filet
            if best[side] is None or conf > best[side][0]:
                best[side] = (conf, foot, (wx, wy), (x1, y1, x2, y2))

        # --- Calcul de la vitesse de chaque joueur ---
        positions = {}
        for side in ("far", "near"):
            if best[side] is None:                   # joueur non detecte sur cette image
                positions[side] = None
                continue
            _, foot, world, box = best[side]
            positions[side] = world
            history[side].append(world)              # memorise la position

            if len(history[side]) > WINDOW:          # assez d'images pour calculer
                p0, p1 = history[side][0], history[side][-1]       # plus ancienne et plus recente
                dist = np.hypot(p1[0] - p0[0], p1[1] - p0[1])      # distance parcourue (m)
                v = dist / (WINDOW / fps)                          # vitesse = distance / temps
                if v <= MAX_HUMAN_SPEED:                           # filtre les valeurs impossibles
                    cur_speed[side] = v
                    max_speed[side] = max(max_speed[side], v)

            # Rectangle au-dessus de la tete + vitesse en km/h (m/s x 3.6)
            color = (0, 200, 255) if side == "far" else (255, 120, 0)
            draw_player_marker(frame, box, color, cur_speed[side] * 3.6)

        # --- Affichage des vitesses max en haut + mini-carte ---
        cv2.putText(frame,
                    f"Vmax loin: {max_speed['far'] * 3.6:.1f} km/h | "
                    f"Vmax proche: {max_speed['near'] * 3.6:.1f} km/h",
                    (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        draw_minimap(frame, positions)

        # --- Sauvegarde et affichage ---
        out.write(frame)                             # ajoute l'image a la video de sortie
        cv2.imshow("Tennis Analyzer (q pour quitter)",
                   cv2.resize(frame, None, fx=view_scale, fy=view_scale))
        if cv2.waitKey(1) & 0xFF == ord("q"):        # touche q = arreter
            break

    # --- Fin : fermeture et resume ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\n=== Resume ===")
    print(f"Joueur loin  : vitesse max {max_speed['far'] * 3.6:.1f} km/h")
    print(f"Joueur proche: vitesse max {max_speed['near'] * 3.6:.1f} km/h")
    print("Video annotee : output_tennis.mp4")


# ---------------------------------------------------------------
# POINT D'ENTREE : s'execute quand on lance "python tennis_analyzer.py video.mp4"
# ---------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:                            # pas de video donnee
        sys.exit("Usage : python tennis_analyzer.py match.mp4")
    main(sys.argv[1])                                # sys.argv[1] = nom de la video