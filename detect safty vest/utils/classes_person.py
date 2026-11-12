"""Classify each person: worker or not worker (by clothing)."""

import math

import cv2
import numpy as np

# Roles
WORKER = "worker"            # has a helmet and/or a vest -> must wear both
NOT_WORKER = "not worker"    # no helmet and no vest -> not checked
OUTSIDE = "outside"          # outside the site zone -> not checked

REQUIRED_PPE = {
    WORKER: ("helmet", "vest"),
    NOT_WORKER: (),
    OUTSIDE: (),
}

# Activities (kept for compatibility)
MOVING = "moving"
STILL = "still"
UNKNOWN = "unknown"

WATCH_MOVE_PIXELS = 40

HELMET_MAX_REL_Y = 0.40


# ---------- Geometry ----------

def foot_point(person_box):
    x1, y1, x2, y2 = person_box
    return ((x1 + x2) // 2, y2)


def is_inside_site(person_box, zone):
    """True if the feet are inside the site polygon (None = whole image)."""
    if zone is None:
        return True

    polygon = np.array(zone, dtype=np.int32)
    x, y = foot_point(person_box)
    return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def find_item(person_box, items, min_rel_y=0.0, max_rel_y=1.0):
    """Return the helmet/vest box that belongs to this person, or None."""
    px1, py1, px2, py2 = person_box
    height = py2 - py1

    if height <= 0:
        return None

    for x1, y1, x2, y2 in items:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        inside = px1 <= cx <= px2 and py1 <= cy <= py2
        rel_y = (cy - py1) / height

        if inside and min_rel_y <= rel_y <= max_rel_y:
            return (x1, y1, x2, y2)

    return None


# ---------- Clothing colour (optional, for information) ----------

def dominant_color(frame, box):
    """Main colour of a box: white, yellow, orange, red, blue or other."""
    x1, y1, x2, y2 = box
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]

    if crop.size == 0:
        return "other"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    masks = {
        "white": (s < 50) & (v > 170),
        "orange": (h >= 5) & (h < 20) & (s > 120) & (v > 120),
        "yellow": (h >= 20) & (h < 50) & (s > 100) & (v > 120),
        "red": ((h < 5) | (h > 170)) & (s > 120) & (v > 90),
        "blue": (h >= 90) & (h < 130) & (s > 100) & (v > 60),
    }

    counts = {name: m.mean() for name, m in masks.items()}
    best = max(counts, key=counts.get)

    return best if counts[best] > 0.25 else "other"


# ---------- Activity (kept for compatibility) ----------

class ActivityTracker:
    """Remember where each tracked person was, to know if they move."""

    def __init__(self):
        self.anchor = {}

    def update(self, track_id, center, t):
        if track_id is None:
            return UNKNOWN

        if track_id not in self.anchor:
            self.anchor[track_id] = (center, t)
            return STILL

        (ax, ay), _ = self.anchor[track_id]
        distance = math.hypot(center[0] - ax, center[1] - ay)

        if distance > WATCH_MOVE_PIXELS:
            self.anchor[track_id] = (center, t)
            return MOVING

        return STILL


# ---------- Role ----------

def classify_person(frame, person_box, helmets, vests, zone=None,
                    track_id=None, tracker=None, t=0.0):
    """Return (role, activity, helmet_color, vest_color)."""

    x1, y1, x2, y2 = person_box
    center = ((x1 + x2) // 2, (y1 + y2) // 2)

    activity = tracker.update(track_id, center, t) if tracker else UNKNOWN

    # 1. Outside the site: ignored
    if not is_inside_site(person_box, zone):
        return OUTSIDE, activity, None, None

    helmet = find_item(person_box, helmets, 0.0, HELMET_MAX_REL_Y)
    vest = find_item(person_box, vests, 0.15, 0.75)

    helmet_color = dominant_color(frame, helmet) if helmet else None
    vest_color = dominant_color(frame, vest) if vest else None

    # 2. At least a helmet OR a vest (or both) -> worker
    if helmet is not None or vest is not None:
        return WORKER, activity, helmet_color, vest_color

    # 3. Nothing -> not worker
    return NOT_WORKER, activity, None, None


def missing_ppe(role, helmet_color, vest_color):
    """List the missing equipment for this role."""
    missing = []

    if "helmet" in REQUIRED_PPE[role] and helmet_color is None:
        missing.append("helmet")
    if "vest" in REQUIRED_PPE[role] and vest_color is None:
        missing.append("vest")

    return missing