import os
import sys
import json
import time
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
os.chdir(ROOT)

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils.classes_person import (
    ActivityTracker, classify_person, missing_ppe,
    WORKER, NOT_WORKER, OUTSIDE,
)
from utils.detect_casque import get_helmets
from utils.detect_vest import get_vests
from utils.draw import draw_person

# ---------- Réglages ----------
VIDEO_PATH = "Recording 2026-10-28 212554.mp4"
OUTPUT_PATH = "output_safety.mp4"
# MODE = "live": affichage en direct a la VITESSE NORMALE (detection en arriere-plan,
#                 les images sont sautees si le PC est lent). <-- recommande
# MODE = "save": traite chaque image et ecrit output_safety.mp4 a la vitesse originale
MODE = "live"
SAFETY_MODEL = "best.pt"
PERSON_MODEL = "yolo11s.pt"
ZONE_PATH = "site_zone.json"

CONF_PERSON = 0.2
CONF_TILE = 0.25                 # confiance sur les tuiles (petites personnes)
CONF_PPE = 0.25
IMGSZ_PERSON = 960               # image entière
IMGSZ_PPE = 960
DETECT_EVERY = 3                 # YOLO 1 frame sur 3
VIOLATION_SECONDS = 2

# --- Vue aérienne : personnes minuscules -> détection par tuiles ---
USE_TILES = False
TILE = 640                       # taille d'une tuile (pixels de la vidéo)
TILE_OVERLAP = 0.25
TILE_IMGSZ = 960                 # les tuiles sont agrandies avant YOLO
MIN_PERSON_H = 8                 # en dessous : ignoré
SMALL_PERSON_H = 60              # en dessous : trop petit pour casque/gilet -> test couleur
MIN_CHECK_FRAC = 0.20             # personne < 20 % de la hauteur image : ignoree (pas de fausse alerte)
HIVIS_RATIO = 0.02               # part de pixels orange/jaune fluo pour dire "gilet"

DUP_OVERLAP = 0.6
SHOW_PREVIEW = True
DISPLAY_SIZE = (960, 540)

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (160, 160, 160)
ORANGE = (0, 165, 255)

use_gpu = torch.cuda.is_available()
device = 0 if use_gpu else "cpu"
print("GPU :", use_gpu)

# ---------- Chargement ----------
print("Chargement des modèles...")
safety_model = YOLO(SAFETY_MODEL)
person_model = YOLO(PERSON_MODEL)

zone = None
if os.path.exists(ZONE_PATH):
    with open(ZONE_PATH, "r") as f:
        zone = json.load(f)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Vidéo : {width}x{height} | {fps:.1f} fps | {total} frames")

writer = cv2.VideoWriter(
    OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)


# ---------- Boîtes en double (NMS par IoU + inclusion de boîtes de taille proche) ----------
def remove_duplicates(dets):
    """dets: liste de (box, conf). Garde une seule boîte par personne."""
    dets = sorted(dets, key=lambda d: d[1], reverse=True)
    kept = []
    for box, conf in dets:
        x1, y1, x2, y2 = box
        area = max(1, (x2 - x1) * (y2 - y1))
        duplicate = False
        for kb, _ in kept:
            ix1, iy1 = max(x1, kb[0]), max(y1, kb[1])
            ix2, iy2 = min(x2, kb[2]), min(y2, kb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            kept_area = max(1, (kb[2] - kb[0]) * (kb[3] - kb[1]))
            iou = inter / (area + kept_area - inter)
            small, big = min(area, kept_area), max(area, kept_area)
            # une petite boite DANS une grande boite n'est PAS un doublon
            if iou > 0.5 or (inter / small > 0.8 and small / big > 0.5):
                duplicate = True
                break
        if not duplicate:
            kept.append((box, conf))
    return kept


# ---------- Détection des personnes (image entière + tuiles) ----------
BIG_FRAC = 0.25        # une personne plus haute que 25 % de l'image = plan rapproché
BIG_CONF = 0.4         # une grande boite doit etre sure, sinon c'est une fausse detection


def detect_persons(frame):
    H, W = frame.shape[:2]

    r = person_model(frame, classes=[0], conf=CONF_PERSON, iou=0.5,
                     imgsz=IMGSZ_PERSON, device=device, verbose=False)[0]
    full = [(tuple(map(int, b.xyxy[0])), float(b.conf[0])) for b in r.boxes]

    # Grandes boites peu sures (ex. tout le chantier vu du ciel) : supprimees
    full = [(b, c) for b, c in full if (b[3] - b[1]) <= BIG_FRAC * H or c >= BIG_CONF]
    # Boite geante (plus de 45 % de la surface de l'image) : acceptee seulement si tres sure
    full = [(b, c) for b, c in full
            if (b[2] - b[0]) * (b[3] - b[1]) <= 0.45 * W * H or c >= 0.7]

    # Plan rapproche : une grande personne sure est visible -> pas de tuiles
    ground_scene = any((b[3] - b[1]) > BIG_FRAC * H for b, _ in full)

    dets = list(full)

    if USE_TILES and not ground_scene:
        step = int(TILE * (1 - TILE_OVERLAP))
        xs = list(range(0, max(W - TILE, 0) + 1, step))
        ys = list(range(0, max(H - TILE, 0) + 1, step))
        if xs[-1] + TILE < W:
            xs.append(max(W - TILE, 0))
        if ys[-1] + TILE < H:
            ys.append(max(H - TILE, 0))

        for y0 in ys:
            for x0 in xs:
                tile = frame[y0:y0 + TILE, x0:x0 + TILE]
                rt = person_model(tile, classes=[0], conf=CONF_TILE, iou=0.5,
                                  imgsz=TILE_IMGSZ, device=device,
                                  verbose=False)[0]
                for b in rt.boxes:
                    a, bb, c, d = map(int, b.xyxy[0])
                    w, h = c - a, d - bb
                    # personne vue du ciel : petite, et plutot haute que large
                    if h > BIG_FRAC * H or h < MIN_PERSON_H or w <= 0:
                        continue
                    if not (0.7 <= h / w <= 4.5):
                        continue
                    dets.append(((a + x0, bb + y0, c + x0, d + y0),
                                 float(b.conf[0])))

    return [b for b, _ in remove_duplicates(dets)]


# ---------- Petit suivi par centroïdes (identifiants stables) ----------
class SimpleTracker:
    def __init__(self, max_missed=10):
        self.objs = {}          # id -> [cx, cy, missed]
        self.next_id = 1
        self.max_missed = max_missed

    def update(self, boxes):
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[3] - b[1]) for b in boxes]
        used = set()
        ids = [None] * len(boxes)

        for i, (cx, cy, h) in enumerate(centers):
            best, best_d = None, max(30, 1.5 * h)
            for oid, (ox, oy, _) in self.objs.items():
                if oid in used:
                    continue
                d = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = oid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
            used.add(best)
            self.objs[best] = [cx, cy, 0]
            ids[i] = best

        for oid in list(self.objs):
            if oid not in used:
                self.objs[oid][2] += 1
                if self.objs[oid][2] > self.max_missed:
                    del self.objs[oid]
        return ids


# ---------- Gilet fluo par la couleur (personnes très petites) ----------
def hivis_ratio(frame, box):
    x1, y1, x2, y2 = box
    px, py = int(0.3 * (x2 - x1)) + 2, int(0.3 * (y2 - y1)) + 2   # boite elargie
    crop = frame[max(y1 - py, 0):y2 + py, max(x1 - px, 0):x2 + px]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (5, 120, 120), (20, 255, 255))
    yellow = cv2.inRange(hsv, (22, 100, 150), (40, 255, 255))
    mask = cv2.bitwise_or(orange, yellow)
    return float(mask.sum() / 255) / (crop.shape[0] * crop.shape[1])


# ---------- Casques / gilets des personnes moyennes ----------
def detect_ppe_on_crops(frame, persons):
    """Re-détecte casque/gilet sur un crop agrandi des petites personnes."""
    extra_h, extra_v = [], []
    H, W = frame.shape[:2]
    n = 0
    for (x1, y1, x2, y2), tid in persons:
        ph = y2 - y1
        if ph < SMALL_PERSON_H or ph >= 200:
            continue
        if n >= 6:
            break
        n += 1
        pad = int(0.15 * ph)
        cx1, cy1 = max(x1 - pad, 0), max(y1 - pad, 0)
        cx2, cy2 = min(x2 + pad, W), min(y2 + pad, H)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        scale = 480 / crop.shape[0]
        up = cv2.resize(crop, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC)
        r = safety_model(up, conf=CONF_PPE, imgsz=480,
                         device=device, verbose=False)[0]
        for (a, b, c, d) in get_helmets(r):
            extra_h.append((int(a / scale) + cx1, int(b / scale) + cy1,
                            int(c / scale) + cx1, int(d / scale) + cy1))
        for (a, b, c, d) in get_vests(r):
            extra_v.append((int(a / scale) + cx1, int(b / scale) + cy1,
                            int(c / scale) + cx1, int(d / scale) + cy1))
    return extra_h, extra_v


tracker = ActivityTracker()
simple_tracker = SimpleTracker()
bad_since = {}


def detect_all(frame):
    """Toutes les detections sur une image."""
    boxes = detect_persons(frame)
    ids = simple_tracker.update(boxes)
    persons = list(zip(boxes, ids))

    res_s = safety_model(frame, conf=CONF_PPE, imgsz=IMGSZ_PPE,
                         device=device, verbose=False)[0]
    helmets = get_helmets(res_s)
    vests = get_vests(res_s)
    extra_h, extra_v = detect_ppe_on_crops(frame, persons)
    return persons, helmets + extra_h, vests + extra_v


def annotate(frame, t, persons, helmets, vests):
    if zone is not None:
        cv2.polylines(frame, [np.array(zone, dtype=np.int32)], True, (255, 255, 0), 2)

    workers_total = 0
    workers_bad = 0
    active = set()

    for box, tid in persons:
        x1, y1, x2, y2 = box
        ph = y2 - y1
        if ph < MIN_PERSON_H:
            continue

        # Personne trop petite / trop lointaine : on ne la juge pas (evite les fausses alertes)
        if ph < MIN_CHECK_FRAC * frame.shape[0]:
            continue

        role, activity, helmet_color, vest_color = classify_person(
            frame, box, helmets, vests, zone, tid, tracker, t
        )

        # Regle : au moins un casque OU un gilet = worker, sinon not worker
        if role != OUTSIDE:
            role = WORKER if (helmet_color or vest_color) else NOT_WORKER

        if role == OUTSIDE or role == NOT_WORKER:
            cv2.rectangle(frame, (x1, y1), (x2, y2), GRAY, 1)
            if role == NOT_WORKER:
                cv2.putText(frame, "not worker", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
            continue

        missing = missing_ppe(role, helmet_color, vest_color)
        active.add(tid)
        if missing:
            bad_since.setdefault(tid, t)
            violation = (t - bad_since[tid]) >= VIOLATION_SECONDS
        else:
            bad_since.pop(tid, None)
            violation = False

        if role == WORKER:
            workers_total += 1
            if violation:
                workers_bad += 1

        label = (f"{role}: no " + " + no ".join(missing)) if violation else role
        draw_person(frame, box, label, not violation)

    for k in list(bad_since):
        if k not in active:
            del bad_since[k]

    cv2.rectangle(frame, (10, 10), (400, 100), BLACK, -1)
    cv2.putText(frame, f"Workers: {workers_total}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    cv2.putText(frame, f"Not compliant: {workers_bad}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                RED if workers_bad > 0 else GREEN, 2)


# =====================================================================
# MODE "live" : video a vitesse normale, detection dans un thread
# =====================================================================
def run_live():
    lock = threading.Lock()
    pending = {"frame": None}
    results = {"data": ([], [], [])}
    stop = {"flag": False}

    def detector_loop():
        while not stop["flag"]:
            with lock:
                f = pending["frame"]
                pending["frame"] = None
            if f is None:
                time.sleep(0.005)
                continue
            out = detect_all(f)
            with lock:
                results["data"] = out

    threading.Thread(target=detector_loop, daemon=True).start()

    frame_idx = 0
    start = time.time()
    while True:
        # En retard ? on saute des images pour garder la vitesse normale
        target = int((time.time() - start) * fps)
        while frame_idx < target:
            if not cap.grab():
                break
            frame_idx += 1

        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t = frame_idx / fps

        with lock:
            pending["frame"] = frame.copy()
            persons, helmets, vests = results["data"]

        annotate(frame, t, persons, helmets, vests)
        cv2.imshow("Safety detection (Q pour quitter)",
                   cv2.resize(frame, DISPLAY_SIZE))

        ahead = (frame_idx / fps) - (time.time() - start)
        if cv2.waitKey(max(1, int(ahead * 1000))) & 0xFF == ord("q"):
            break
    stop["flag"] = True


# =====================================================================
# MODE "save" : chaque image traitee, video finale a vitesse originale
# =====================================================================
def run_save():
    persons, helmets, vests = [], [], []
    frame_idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t = frame_idx / fps
        if DETECT_EVERY == 1 or frame_idx % DETECT_EVERY == 1:
            persons, helmets, vests = detect_all(frame)
        annotate(frame, t, persons, helmets, vests)
        writer.write(frame)

        if frame_idx % 10 == 0:
            speed = frame_idx / (time.time() - t0)
            eta = (total - frame_idx) / speed if speed > 0 else 0
            print(f"{frame_idx}/{total} frames | {speed:.1f} fps | reste ~{eta/60:.1f} min")
        if SHOW_PREVIEW:
            cv2.imshow("Saving... (Q pour quitter)", cv2.resize(frame, DISPLAY_SIZE))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    print(f"Termine : {OUTPUT_PATH}")


if MODE == "live":
    run_live()
else:
    run_save()

cap.release()
writer.release()
cv2.destroyAllWindows()