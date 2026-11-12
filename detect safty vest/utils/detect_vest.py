VEST_NAMES = {"vest"}

VEST_MIN_REL_Y=0.15
VEST_MAX_REL_Y=0.75

def get_vests(result):
    vests=[]

    for box in result.boxes:
        name=result.names[int(box.cls[0])].lower()
        if  name in VEST_NAMES:
            vests.append(tuple(map(int,box.xyxy[0])))   
    return vests  
def has_vest(person_box, vests):
    """True if one vest is on this person's body."""
    px1, py1, px2, py2 = person_box
    height = py2 - py1

    if height <= 0:
        return False

    for vx1, vy1, vx2, vy2 in vests:
        cx = (vx1 + vx2) / 2
        cy = (vy1 + vy2) / 2

        inside = px1 <= cx <= px2 and py1 <= cy <= py2
        rel_y = (cy - py1) / height

        if inside and VEST_MIN_REL_Y <= rel_y <= VEST_MAX_REL_Y:
            return True

    return False          



