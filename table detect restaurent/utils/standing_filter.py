THIGH_TO_TORSO=0.4
KNEE_CONFIDENCE=0.5
SAME_PERSON=0.5

def is_standing(keypoints):
    points=keypoints.xy[0].tolist(
        confidence=keypoints.conf[0].tolist
    )