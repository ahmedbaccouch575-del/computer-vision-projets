
import cv2
import numpy as np

TINT = 0.4


def people_at_table(points, centers):
    polygon = np.array(points, np.int32)
    return sum(
        cv2.pointPolygonTest(polygon, center, False) >= 0
        for center in centers
    )


def draw_table_tops(frame, table_tops, colors):
    overlay = frame.copy()

    for name, points in table_tops.items():
        table_top = np.array(points, np.int32)

        color = colors.get(name, (0, 255, 0))

        cv2.fillPoly(overlay, [table_top], color)
        cv2.polylines(frame, [table_top], True, color, 2)

    return cv2.addWeighted(overlay, TINT, frame, 1 - TINT, 0)
