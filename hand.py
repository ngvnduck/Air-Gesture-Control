#!/usr/bin/python3

import cv2
import mediapipe as mp
import numpy as np
from picamera2 import Picamera2

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Reduce confidence for speed
mp_drawing = mp.solutions.drawing_utils

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))  # Set RGB format for compatibility
picam2.start()

prev_landmarks = None

def detect_motion(landmarks, prev_landmarks):
    if prev_landmarks is None:
        return None

    motions = {}
    for i, landmark in enumerate(landmarks):
        prev_lm = prev_landmarks[i]
        dx = landmark[0] - prev_lm[0]
        dy = landmark[1] - prev_lm[1]

        if abs(dx) > abs(dy):
            motions[i] = 'Right' if dx > 0 else 'Left'
        else:
            motions[i] = 'Down' if dy > 0 else 'Up'
    return motions

frame_count = 0
while True:
    # Capture image from Pi Camera
    img = picam2.capture_array()

    # Check if the image has the correct shape
    if img.ndim != 3 or img.shape[2] != 3:
        print("Warning: Image does not have 3 channels. Converting to BGR.")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert if necessary

    # Process every frame for testing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe processing
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand landmarks into a list of (x, y) tuples
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

            # Detect motion of landmarks compared to previous frame
            motions = detect_motion(landmarks, prev_landmarks)

            # For now, we just display motions for landmark 8 (index finger tip)
            if motions and 8 in motions:
                motion = motions[8]
                cv2.putText(img, f"Index Finger: {motion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Update previous landmarks
            prev_landmarks = landmarks

    # Display the result
    cv2.imshow("Camera", img)

    # Break loop on 'Esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
