import cv2
import numpy as np


def generate_triangles(points, width, height):

    rect = (0, 0, width, height)

    subdiv = cv2.Subdiv2D(rect)

    for p in points:
        subdiv.insert((int(p[0]), int(p[1])))

    triangle_list = subdiv.getTriangleList()

    triangles = []

    for t in triangle_list:

        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        triangles.append([pt1, pt2, pt3])

    return triangles
