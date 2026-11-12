MAX_OVERLAP = 0.7


def overlap(box_a, box_b):
    # How much of the smaller box is covered by the other box, from 0 to 1
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    width = min(ax2, bx2) - max(ax1, bx1)
    height = min(ay2, by2) - max(ay1, by1)
    if width <= 0 or height <= 0:
        return 0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return (width * height) / min(area_a, area_b)


def remove_duplicates(boxes):
    corners = boxes.xyxy.tolist()
    confidences = boxes.conf.tolist()

    # Sort box indexes from highest to lowest confidence
    order = sorted(range(len(corners)), key=lambda i: confidences[i], reverse=True)

    kept = []
    for i in order:
        if all(overlap(corners[i], corners[k]) < MAX_OVERLAP for k in kept):
            kept.append(i)

    return [boxes[i] for i in kept]