HELMET_NAMES = {"blue helmet", "red helmet", "white helmet", "yellow helmet"}
HELMET_MAX_REL_Y=0.40


def get_helmets(result):
    helmets=[]
    for box in result.boxes:
        name=result.names[int(box.cls[0])].lower()

        if name in HELMET_NAMES :
            helmets.append(tuple(map(int,box.xyxy[0])))

    return helmets       
def has_helmet(person_box, helmets):
    """True if one helmet is on this person's head."""
    px1, py1, px2, py2 = person_box
    height = py2 - py1

    if height <= 0:
        return False

    for hx1, hy1, hx2, hy2 in helmets:
        cx = (hx1 + hx2) / 2
        cy = (hy1 + hy2) / 2

        inside = px1 <= cx <= px2 and py1 <= cy <= py2
        rel_y = (cy - py1) / height

        if inside and rel_y <= HELMET_MAX_REL_Y:
            return True

    return False        
