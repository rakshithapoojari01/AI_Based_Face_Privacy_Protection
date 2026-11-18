import cv2
import numpy as np


def move_point(point, dx, dy):

    return (
        int(point[0] + dx),
        int(point[1] + dy)
    )
def deform_region(points, indices, dx, dy):
    
    new_points = points.copy()

    for idx in indices:

        if idx < len(new_points):

            x, y = new_points[idx]

            new_points[idx] = (
                x + dx,
                y + dy
            )

    return new_points