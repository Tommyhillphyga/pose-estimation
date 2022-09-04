import numpy as np
import math
import cv2



def findAngle(image,kpts,p1,p2,p3, draw=True): # takes three points and return the angle between them
    coord = []
    no_kpt = len(kpts) // 3
    for i in range (no_kpt):
        cx, cy = kpts[3*i], kpts[3*i + 1]
        conf = kpts[3 * i + 2]
        coord.append([i, cx, cy, conf])

    points = (p1,p2,p3)
    # angle = 0
    
    # if (coord[p1][3] > 0.5 and coord[p2][3] > 0.5 and coord[p3][3] > 0.5):
    #   print([coord[p1][3], coord[p2][3], coord[p3][3]])


    #Get landmarks
    x1, y1 = coord[p1][1:3]
    x2, y2 = coord[p2][1:3]
    x3, y3 = coord[p3][1:3]

    #calculate the angle
    angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))

    if angle < 0:
        angle+=360

    #draw coordinates
    if draw:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 3)
        cv2.line(image, (int(x3), int(y3)), (int(x2), int(y2)), (255,255,255), 3)
        cv2.circle(image, (int(x1),int(y1)), 10, (255, 255, 255), cv2.FILLED)
        cv2.circle(image, (int(x1),int(y1)), 20, (235, 235, 235), 5)
        cv2.circle(image, (int(x2),int(y2)), 10, (255, 255, 255), cv2.FILLED)
        cv2.circle(image, (int(x2),int(y2)), 20, (235, 235, 235), 5)
        cv2.circle(image, (int(x3),int(y3)), 10, (255, 255, 255), cv2.FILLED)
        cv2.circle(image, (int(x3),int(y3)), 20, (235, 235, 235), 5)
        # cv2.putText(image, str(int(angle)), (int(x2) - 50, int(y2) + 50),
        #                 cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return int(angle)