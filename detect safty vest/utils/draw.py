import cv2


GREEN=(0,255,0)
RED=(0,0,255)
WHITE=(255,255,255)
FONT=cv2.FONT_HERSHEY_SIMPLEX


def draw_person(frame,box,label,is_ok):
    """ draw a person box with a label
    gree=compliant,red=violation"""
    x1,y1,x2,y2=box
    color=GREEN if is_ok else RED
    thickness=2 if is_ok else 3
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,thickness)

    # filled  label background above the box
    (tw,th),base=cv2.getTextSize(label,FONT,0.6,2)
    y_top=max(y1-th-base-6,0)
    cv2.rectangle(frame,(x1,y_top),(x1+tw+8,y_top+th+base+6),color,-1)
    cv2.putText(frame, label, (x1 + 4, y_top + th + 2), FONT, 0.6, WHITE, 2)