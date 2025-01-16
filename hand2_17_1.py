#!/usr/bin/python3

import cv2
import mediapipe as mp
import numpy as np
from picamera2 import Picamera2
from libcamera import Transform
import time

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# Initialize the camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(
	main={"format": "RGB888", "size": (640, 480)},
	controls={"FrameDurationLimits": (33333, 33333)}  # Set FPS to 30
)
transform = Transform(hflip=True)
config["transform"] = transform
picam2.configure(config)
picam2.start()

# Landmarks for fingertips and second joints
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_SECOND_JOINTS = [3, 7, 11, 15, 19]
prev_landmarks = None  # Previous frame's landmarks

# Threshold for motion detection
movement_threshold = 0.008  # Minimum normalized distance

# Counter and delay variables
direction_count = 0
last_direction = None
cooldown = False
cooldown_start_time = 0
cooldown_time = 2.0  # Cooldown time (1 second)


def detect_motion(current_landmarks, prev_landmarks):
	"""Detect finger motion direction, ignoring minor jitter."""
	if prev_landmarks is None:
    	return None

	dx_total, dy_total = 0, 0
	significant_movement = False

	for tip_idx, joint_idx in zip(FINGER_TIPS, FINGER_SECOND_JOINTS):
    	# Get the coordinates of the tip and second joint
    	current_tip = current_landmarks[tip_idx]
    	prev_tip = prev_landmarks[tip_idx]
    	current_joint = current_landmarks[joint_idx]
    	prev_joint = prev_landmarks[joint_idx]

    	# Calculate average movement
    	dx = ((current_tip[0] - prev_tip[0]) + (current_joint[0] - prev_joint[0])) / 2
    	dy = ((current_tip[1] - prev_tip[1]) + (current_joint[1] - prev_joint[1])) / 2

    	# Check for significant movement
    	if abs(dx) > movement_threshold or abs(dy) > movement_threshold:
        	significant_movement = True

    	dx_total += dx
    	dy_total += dy

	if not significant_movement:
    	return None  # No significant movement

	# Determine overall motion direction
	if abs(dx_total) > abs(dy_total):  # Prioritize horizontal motion
    	return "Right" if dx_total > 0 else "Left"
	else:
    	return "Down" if dy_total > 0 else "Up"


while True:
	img = picam2.capture_array()

	# Ensure the image has sufficient color channels
	if img.ndim != 3 or img.shape[2] != 3:
    	print("Warning: Image does not have 3 channels.")
    	continue

	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	result = hands.process(img_rgb)

	# Check cooldown
	if cooldown and (time.time() - cooldown_start_time < cooldown_time):
    	cv2.imshow("Camera", img)
    	if cv2.waitKey(1) & 0xFF == 27:
        	break
    	continue
	else:
    	cooldown = False

	# Display hands if detected
	if result.multi_hand_landmarks:
    	current_landmarks = {}
    	for hand_landmarks in result.multi_hand_landmarks:
        	# Extract fingertip and second joint coordinates
        	for idx in FINGER_TIPS + FINGER_SECOND_JOINTS:
            	lm = hand_landmarks.landmark[idx]
            	current_landmarks[idx] = (lm.x, lm.y)

            	# Draw circles on the landmarks
            	h, w, _ = img.shape
            	cx, cy = int(lm.x * w), int(lm.y * h)
            	cv2.circle(img, (cx, cy), 10, (0, 255, 0), -1)

        	# Detect motion direction
        	direction = detect_motion(current_landmarks, prev_landmarks)

        	# Process only if direction is detected
        	if direction:
            	if direction == last_direction:
                	direction_count += 1
            	else:
                	direction_count = 1  # Reset count on direction change
            	last_direction = direction

            	# Output if the direction count reaches 3
            	if direction_count == 3:
                	print(f"Detected Motion: {direction}")
                	cooldown = True
                	cooldown_start_time = time.time()
                	direction_count = 0

        	# Update previous landmarks
        	prev_landmarks = current_landmarks

	else:
    	prev_landmarks = None  # Reset if no hands detected

	# Display the camera feed
	cv2.imshow("Camera", img)

	# Stop program on 'Esc' key press
	if cv2.waitKey(1) & 0xFF == 27:
    	break

cv2.destroyAllWindows()
