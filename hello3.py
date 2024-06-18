import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize mediapipe and pyautogui
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to convert normalized hand coordinates to screen coordinates
def hand_landmarks_to_screen(landmark, image_width, image_height):
    x = int(landmark.x * image_width)
    y = int(landmark.y * image_height)
    return x, y

# Capture video from the webcam
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Flags and variables for additional actions
thumb_open_prev = False  # Previous state of thumb open
thumb_x_prev = 0  # Previous thumb x position

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    image_height, image_width, _ = image.shape

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract coordinates of fingers
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_x, index_y = hand_landmarks_to_screen(index_finger_tip, image_width, image_height)
            thumb_x, thumb_y = hand_landmarks_to_screen(thumb_tip, image_width, image_height)

            # Left Click
            if thumb_x > index_x + 30 and thumb_open_prev:
                pyautogui.click()

            # Update thumb open/close state
            if thumb_x < index_x - 30:  # Thumb to the left of index finger
                thumb_open_prev = False
            elif thumb_x > index_x + 30:  # Thumb to the right of index finger
                thumb_open_prev = True

            # Move the mouse based on thumb position
            if thumb_open_prev:
                move_x = thumb_x - thumb_x_prev
                move_y = thumb_y - index_y
                pyautogui.move(move_x, move_y)

            thumb_x_prev = thumb_x

    # Display the image
    cv2.imshow('Virtual Mouse', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
