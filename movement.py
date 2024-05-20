# -*- coding: utf-8 -*-
"""
Created on Sun May  5 15:24:22 2024

@author: defne
"""

import cv2
import numpy as np
import math
import time

def calculate_displacement(coord1, coord2):
    # Unpack the coordinates
    x1, y1 = coord1
    x2, y2 = coord2

    # Calculate the displacement
    displacement = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return displacement

# Function to find center of contour
def find_center(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy
    else:
        return None, None
    
def calibrate_distance(observed_width, original_distance_cm):
    
    # Calculate the initial distance between leftmost and rightmost points
    initial_distance_px = observed_width
    
    # Calculate the initial constant multiplier
    initial_constant = original_distance_cm / initial_distance_px
    
    # Set an initial step size for adjusting the constant
    step_size = 0.5
    
    # Set an initial threshold for the difference between calculated and target width
    threshold = 1
    
    # Initialize the current constant multiplier
    current_constant = initial_constant
    
    # Initialize the current calculated width
    current_calculated_width_cm = initial_distance_px * current_constant
    
    # Initialize the number of iterations
    iterations = 0
    
    # Loop until the calculated width matches the target width or max number of iterations reached
    while abs(current_calculated_width_cm - original_distance_cm) > threshold and iterations < 100:
        # Update the current constant by adding or subtracting the step size
        if current_calculated_width_cm < original_distance_cm:
            current_constant += step_size
        else:
            current_constant -= step_size
        
        # Recalculate the current calculated width
        current_calculated_width_cm = initial_distance_px * current_constant
        
        # Increment the number of iterations
        iterations += 1
    
    # Return the calibrated constant
    return current_constant

# Load the video
cap = cv2.VideoCapture(1)  # 0 for the default camera, change to 1 or 2 for additional cameras if available

# Create a mask for green color
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

previous_time = time.time()
total_displacement = 0

calibration = False
prev_observed_width = None
#Adjust the value in here
original_width = 11
p=0
q= [0] * 200
while cap.isOpened():
    
    
    calibrate=False
    ret, frame = cap.read()
    if not ret:
        break
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    if contours:
        # Draw contours
        #print("Contours")
        #cv2.drawContours(frame, contours, -1, (255, 255, 255), 2)
        #multiplier = calibrate_distance(w, 12.5)
        # Find center of the largest contour
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            
            if area >= 1500:
                cv2.drawContours(frame, c, -1, (255, 255, 255), 2)
                #print("Area", area)
                remainder=p%200
                q[remainder]=area
                p=p+1
                meanq=sum(q) / len(q)
                upper=meanq+0.01*meanq
                lower=meanq-0.01*meanq
                # print("last 10 area: ",q)
                # print("Mean of list: ",meanq)
                # print("Lower,Upper:",lower,upper)
                if lower<area and area<upper:
                    calibrate=True
                #(x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
                
                x,y,w,h = cv2.boundingRect(c)
                #drawing rectange around the object
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                
                if not calibration and prev_observed_width is None and calibrate: 
                #or (prev_observed_width is not None and abs(w - prev_observed_width) > 25):
                    
                    print("************* Calibration starts ***********")
                    print("*\n"*20)
                    multiplier = calibrate_distance(w, original_width)
                    calibration = True
                    
                    print("Previously observed width:", prev_observed_width)
                    print("width now:", w)
                    # Update previous observed width
                    prev_observed_width = w
                    print("Origianl width:", 12.5)
                    print("width after the calibration:", multiplier*w)
                    print(q)
                    print("************Calibration complete************. Multiplier:", multiplier)
                    print("*\n"*20)
                
                    
                cx, cy = find_center(c)
                if calibration and cx is not None and cy is not None:
                    # Calculate displacement
                    if 'prev_center' in locals():
                        displacement = calculate_displacement(prev_center, (cx, cy))
                        total_displacement += displacement
                        current_time = time.time()
                        time_elapsed = current_time - previous_time
    
                        if time_elapsed >= 1:
                            # Check if displacement is above 10 in 1 second
                            total_displacement_cm= total_displacement * multiplier
                            print(total_displacement_cm)
                            if total_displacement_cm >= 20:
                                print("Displacement is above 20 cm in 1 second")
                            # Reset variables for next second
                            previous_time = current_time
                            total_displacement = 0
    
                    prev_center = (cx, cy)
    
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            #else:
                #print("Baby Lost")
    #else:
        #print("Baby lost")
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()