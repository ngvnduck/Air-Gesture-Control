#!/usr/bin/python3

import numpy as np
import cv2
import math
from picamera2 import Picamera2

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Variables to store the previous position of the hand
prev_centroid = None
motion_threshold = 20  # Minimum movement to consider as a gesture

while True:
    # Capture image from Pi Camera
    img = picam2.capture_array()

    # Convert image to grayscale and blur it for better thresholding
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (35, 35), 0)

    # Apply binary thresholding to isolate the hand
    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assuming it's the hand)
        largest_contour = max(contours, key=cv2.contourArea)

        # Draw the bounding box around the hand
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Find the centroid of the hand
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Draw a circle at the centroid
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

            # Detect motion based on the difference between the current and previous centroid
            if prev_centroid is not None:
                dx = cx - prev_centroid[0]
                dy = cy - prev_centroid[1]

                # Determine the direction of the movement
                if abs(dx) > abs(dy) and abs(dx) > motion_threshold:
                    if dx > 0:
                        gesture = "Right"
                    else:
                        gesture = "Left"
                elif abs(dy) > motion_threshold:
                    if dy > 0:
                        gesture = "Down"
                    else:
                        gesture = "Up"
                else:
                    gesture = None

                if gesture:
                    cv2.putText(img, f"{gesture} Motion", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            prev_centroid = (cx, cy)
        else:
            prev_centroid = None

    # Display the result
    cv2.imshow("Camera", img)

    # Break loop on 'Esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
