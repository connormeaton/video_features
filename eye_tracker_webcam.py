# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 19:21:18 2020

@author: hp
"""

import cv2
import numpy as np
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
# from head_pose_estimation import *


eye_roll_l = []
eye_roll_r = []


def eye_on_mask(mask, side, shape):
    """
    Create ROI on mask of the size of eyes and also find the extreme points of each eye

    Parameters
    ----------
    mask : np.uint8
        Blank mask to draw eyes on
    side : list of int
        the facial landmark numbers of eyes
    shape : Array of uint32
        Facial landmarks

    Returns
    -------
    mask : np.uint8
        Mask with region of interest drawn
    [l, t, r, b] : list
        left, top, right, and bottommost points of ROI

    """
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1]+points[2][1])//2
    r = points[3][0]
    b = (points[4][1]+points[5][1])//2
    return mask, [l, t, r, b]

def find_eyeball_position(end_points, cx, cy):
    """Find and return the eyeball positions, i.e. left or right or top or normal"""
    x_ratio = (end_points[0] - cx)/(cx - end_points[2])
    y_ratio = (cy - end_points[1])/(end_points[3] - cy)
    text = ''
    if x_ratio > 3 and y_ratio < 1:
        font = cv2.FONT_HERSHEY_SIMPLEX 
        text = 'diagonal up left'
        cv2.putText(img, text, (30, 30), font,  
                   1, (0, 255, 0), 2, cv2.LINE_AA) 
    if x_ratio < .33 and y_ratio < 1:
        font = cv2.FONT_HERSHEY_SIMPLEX 
        text = 'diagonal up right'
        cv2.putText(img, text, (30, 30), font,  
                   1, (0, 255, 0), 2, cv2.LINE_AA) 

    if x_ratio > 3:
        return 1
    elif x_ratio < 0.33:
        return 2
    elif y_ratio < 0.33:
        return 3
    else:
        return 0

def find_eyeroll(end_points, cx, cy):
    """Find and return the eyeball positions, i.e. left or right or top or normal"""
    x_ratio = (end_points[0] - cx)/(cx - end_points[2])
    y_ratio = (cy - end_points[1])/(end_points[3] - cy)
    text = ''
    if x_ratio > 3 and y_ratio < 1:
        print('eyeroll')
        font = cv2.FONT_HERSHEY_SIMPLEX 
        text = 'diagonal up left'
        cv2.putText(img, text, (30, 30), font,  
                   1, (0, 255, 0), 2, cv2.LINE_AA) 
        return True
    if x_ratio < .33 and y_ratio < 1:
        print('eyeroll')
        font = cv2.FONT_HERSHEY_SIMPLEX 
        text = 'diagonal up right'
        cv2.putText(img, text, (30, 30), font,  
                   1, (0, 255, 0), 2, cv2.LINE_AA) 
        return True
    
    else:
        return False


def contouring(thresh, mid, img, end_points, right=False):
    """
    Find the largest contour on an image divided by a midpoint and subsequently the eye position

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image of one side containing the eyeball
    mid : int
        The mid point between the eyes
    img : Array of uint8
        Original Image
    end_points : list
        List containing the exteme points of eye
    right : boolean, optional
        Whether calculating for right eye or left eye. The default is False.

    Returns
    -------
    pos: int
        the position where eyeball is:
            0 for normal
            1 for left
            2 for right
            3 for up

    """
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 255, 0), 2)
        pos = find_eyeball_position(end_points, cx, cy)
        eyeroll = find_eyeroll(end_points, cx, cy)
        return pos
    except:
        pass
    
def process_thresh(thresh):
    """
    Preprocessing the thresholded image

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image to preprocess

    Returns
    -------
    thresh : Array of uint8
        Processed thresholded image

    """
    thresh = cv2.erode(thresh, None, iterations=2) 
    thresh = cv2.dilate(thresh, None, iterations=4) 
    thresh = cv2.medianBlur(thresh, 3) 
    thresh = cv2.bitwise_not(thresh)
    return thresh

def print_eye_pos(img, left, right):
    """
    Print the side where eye is looking and display on image

    Parameters
    ----------
    img : Array of uint8
        Image to display on
    left : int
        Position obtained of left eye.
    right : int
        Position obtained of right eye.

    Returns
    -------
    None.

    """
    # if left == right and left != 0:
    #     text = ''
    #     if left == 1:
    #         print('Looking left')
    #         text = 'Looking left'
    #     elif left == 2:
    #         print('Looking right')
    #         text = 'Looking right'
    #     elif left == 3:
    #         print('Looking up')
    #         text = 'Looking up'
    #     font = cv2.FONT_HERSHEY_SIMPLEX 
        # cv2.putText(img, text, (30, 30), font,  
        #            1, (0, 255, 255), 2, cv2.LINE_AA) 

face_model = get_face_detector()
landmark_model = get_landmark_model()
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)

ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

l = []
r = []
while(True):
    ret, img = cap.read()
    if not ret:
        break

    rects = find_faces(img, face_model)
    
    for rect in rects:
        shape = detect_marks(img, landmark_model, rect)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask, end_points_left = eye_on_mask(mask, left, shape)
        mask, end_points_right = eye_on_mask(mask, right, shape)
  
        mask = cv2.dilate(mask, kernel, 5)
        
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        print(threshold)
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = process_thresh(thresh)
        
        mid = int(mid)
        eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
        eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
        print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)

        ########

    #     marks = detect_marks(img, landmark_model, face)
    #     # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
    #     image_points = np.array([
    #                             marks[30],     # Nose tip
    #                             marks[8],     # Chin
    #                             marks[36],     # Left eye left corner
    #                             marks[45],     # Right eye right corne
    #                             marks[48],     # Left Mouth corner
    #                             marks[54]      # Right mouth corner
    #                         ], dtype="double")
    #     dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    #     (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
        
        
    #     # Project a 3D point (0, 0, 1000.0) onto the image plane.
    #     # We use this to draw a line sticking out of the nose
        
    #     (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
    #     # for p in image_points:
    #     #     cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        
        
    #     p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    #     p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    #     x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

    #     cv2.line(img, p1, p2, (0, 255, 255), 2)
    #     cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
    #     # for (x, y) in marks:
    #     #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
    #     # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
    #     try:
    #         m = (p2[1] - p1[1])/(p2[0] - p1[0])
    #         ang1 = int(math.degrees(math.atan(m)))
    #     except:
    #         ang1 = 90
            
    #     try:
    #         m = (x2[1] - x1[1])/(x2[0] - x1[0])
    #         ang2 = int(math.degrees(math.atan(-1/m)))
    #     except:
    #         ang2 = 90
            
    #         # print('div by zero error')
    #     if ang1 >= 48:
    #         print('Head down')
    #         cv2.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
    #     elif ang1 <= -48:
    #         print('Head up')
    #         cv2.putText(img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)
            
    #     if ang2 >= 48:
    #         print('Head right')
    #         cv2.putText(img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
    #     elif ang2 <= -48:
    #         print('Head left')
    #         cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)
        
    #     cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
    #     cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
    # cv2.imshow('img', img)



    

    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()


    
