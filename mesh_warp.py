import cv2
import numpy as np


def move_points(points, shift_x=0, shift_y=0):

    moved = []

    for x, y in points:

        moved.append([
            x + shift_x,
            y + shift_y
        ])

    return moved


def warp_triangle(img, t1, t2):

    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = []
    t2_rect = []

    for i in range(3):

        t1_rect.append((
            t1[i][0] - r1[0],
            t1[i][1] - r1[1]
        ))

        t2_rect.append((
            t2[i][0] - r2[0],
            t2[i][1] - r2[1]
        ))

    img_rect = img[
        r1[1]:r1[1]+r1[3],
        r1[0]:r1[0]+r1[2]
    ]

    size = (r2[2], r2[3])

    warp_mat = cv2.getAffineTransform(
        np.float32(t1_rect),
        np.float32(t2_rect)
    )

    warped = cv2.warpAffine(
        img_rect,
        warp_mat,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    mask = np.zeros(
        (r2[3], r2[2], 3),
        dtype=np.float32
    )

    cv2.fillConvexPoly(
        mask,
        np.int32(t2_rect),
        (1.0,1.0,1.0)
    )

    img[
        r2[1]:r2[1]+r2[3],
        r2[0]:r2[0]+r2[2]
    ] = img[
        r2[1]:r2[1]+r2[3],
        r2[0]:r2[0]+r2[2]
    ]*(1-mask)+warped*mask

    return img
