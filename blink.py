# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:24:13 2024

@author: defne
"""

import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
# variables 
frame_counter =0
CEF_COUNTER =0  #consecutive frames are counted
TOTAL_BLINKS =0
# constants
CLOSED_EYES_FRAME =3 #threshold for consequtive closed_eyes frames
FONTS =cv.FONT_HERSHEY_COMPLEX

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh
# camera object 
#camera = cv.VideoCapture(0)

# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio 

# Eyes Extrctor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # getting the dimension of image 
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color 
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask 
    # cv.imshow('mask', mask)
    
    # draw eyes image on mask, where white shape is 
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys 
    # cv.imshow('eyes draw', eyes)
    eyes[mask==0]=155
    
    # getting minium and maximum x and y  for right and left eyes 
    # For Right Eye 
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes 
    return cropped_right, cropped_left


# creating pixel counter function 
def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part 
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list 
    max_index = eye_parts.index(max(eye_parts))
    pos_eye ='' 
    if max_index==0:
        pos_eye="RIGHT"
        color=[utils.BLACK, utils.GREEN]
    elif max_index==1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index ==2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye="Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color
#For video case use this:
video_path = 'C:\\Users\\defne\\Desktop\\2023-2024SpringSemester\\EE494\\blink_detection\\cute_baby_aslan.mp4'
cap = cv.VideoCapture(video_path)

#For camera output case use this
#Use cv.VideoCapture(1) for external camera use
camera = cv.VideoCapture(0)
# Constants
desired_fc = 4  #Adjusting this value
awake_threshold = 40 # Threshold for determining the awake state
sleep_threshold = 40  # Threshold for determining the sleeping state

# Variables
awake_counter = 0
sleep_counter = 0

if (camera.isOpened()== False): 
    print("Error opening video file") 

cv.namedWindow('Frame', cv.WINDOW_NORMAL)
cv.setWindowProperty('Frame', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
# starting Video loop here.

sleep = False
awake = False

while(camera.isOpened()):
    
    frame_counter +=1 # frame counter
    #Capture each frame
    ret, frame = camera.read()
    if not ret:
        break #no more frames break
    
    if (ret == True) and (frame_counter % desired_fc == 0):
        
        # starting time here 
        start_time = time.time()
        
        with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
            frame_height, frame_width= frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results  = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, results, False)
                ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK)
                
                if ratio >4.2: # the eyes are close
                    CEF_COUNTER +=1
                    if (sleep_counter < 99): 
                        sleep_counter += 1
                    """
                    if(awake_counter>0):
                        awake_counter -= 1
                    """
                    utils.colorBackgroundText(frame,  'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.PINK, pad_x=6, pad_y=6, )
                    
                    if sleep_counter > sleep_threshold:
                        awake_counter = 0
                        sleep = True
                        awake = False                    
                    
                else: # the eyes are open
                    if awake_counter < 99:
                        awake_counter += 1
                    """
                    if(sleep_counter > 0):
                        sleep_counter -= 1
                    """
                    #blinking
                    if CEF_COUNTER>CLOSED_EYES_FRAME: #eyes have been closed for a certain duration and total_blink is incremented
                        TOTAL_BLINKS +=1
                        CEF_COUNTER =0
                        
                    
                    if awake_counter > awake_threshold :
                        sleep_counter =0
                        sleep = False
                        awake = True
                        
                # Display counters
                utils.colorBackgroundText(frame, f'Sleep Counter: {sleep_counter}', FONTS, 0.7, (30, 200), 2)
                utils.colorBackgroundText(frame, f'Awake Counter: {awake_counter}', FONTS, 0.7, (30, 250), 2)

                utils.colorBackgroundText(frame,  f'Total Number of Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
                
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.PINK, 1, cv.LINE_AA)
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.PINK, 1, cv.LINE_AA)
            
                if  sleep:
                    utils.colorBackgroundText(frame,  'Sleeping', FONTS, 1.7, (int(frame_height/2), 200), 2, utils.PINK, pad_x=6, pad_y=6, )
                    #print('Sleeping')
                elif awake:
                    utils.colorBackgroundText(frame,  'Awake', FONTS, 1.7, (int(frame_height/2), 200), 2, utils.PINK, pad_x=6, pad_y=6, )
                    #print('Awake')
        # calculating  frame per seconds FPS
        fps = 1 / (time.time()-start_time)

        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.1, textThickness=2)
        #Dispaly the resulting freame
        cv.imwrite(f'img/frame_{frame_counter}.png', frame)
        cv.imshow('Frame', frame)
        
        key = cv.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break

# Weighted sum output
#sleep_state = blink_sleep or cry or (movement and word):