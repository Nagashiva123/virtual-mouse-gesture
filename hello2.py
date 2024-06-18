#pip install opencv-python mediapipe pyautogui

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
thumb_prev_close = False  # Previous state of thumb being close to index finger
scroll_direction = 0  # 0 for no scrolling, 1 for scroll up, -1 for scroll down
right_click_prev = False  # Previous state of right click gesture
play_pause_prev = False  # Previous state of play/pause gesture
play_pause_time = time.time()  # Time to debounce play/pause

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

            # Extract coordinates of the index finger tip and thumb tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_x, index_finger_y = hand_landmarks_to_screen(index_finger_tip, image_width, image_height)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_x, thumb_y = hand_landmarks_to_screen(thumb_tip, image_width, image_height)

            # Extract coordinates for right-click gesture (middle finger tip)
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_finger_x, middle_finger_y = hand_landmarks_to_screen(middle_finger_tip, image_width, image_height)

            # Map hand coordinates to screen coordinates
            screen_x = np.interp(index_finger_x, (0, image_width), (0, screen_width))
            screen_y = np.interp(index_finger_y, (0, image_height), (0, screen_height))

            # Move the mouse to the index finger tip position
            pyautogui.moveTo(screen_x, screen_y)

            # Check if the thumb and index finger are close to each other for a click gesture
            distance = np.hypot(index_finger_x - thumb_x, index_finger_y - thumb_y)
            if distance < 30:
                if not thumb_prev_close:
                    pyautogui.click()
                thumb_prev_close = True
            else:
                thumb_prev_close = False

            # Volume Control
            if thumb_prev_close:
                if thumb_y < index_finger_y:  # Thumb above index finger
                    pyautogui.hotkey('volumeup')  # Volume Up
                elif thumb_y > index_finger_y:  # Thumb below index finger
                    pyautogui.hotkey('volumedown')  # Volume Down

            # Scrolling
            if thumb_x < index_finger_x - 20:  # Thumb to the left of index finger
                scroll_direction = 1  # Scroll Up
            elif thumb_x > index_finger_x + 20:  # Thumb to the right of index finger
                scroll_direction = -1  # Scroll Down
            else:
                scroll_direction = 0  # No Scrolling

            # Right-click Gesture
            right_click_distance = np.hypot(middle_finger_x - thumb_x, middle_finger_y - thumb_y)
            if right_click_distance < 30:
                if not right_click_prev:
                    pyautogui.click(button='right')
                right_click_prev = True
            else:
                right_click_prev = False

            # Play/Pause Gesture
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            pinky_x, pinky_y = hand_landmarks_to_screen(pinky_tip, image_width, image_height)
            play_pause_distance = np.hypot(index_finger_x - pinky_x, index_finger_y - pinky_y)
            if play_pause_distance < 30:
                current_time = time.time()
                if current_time - play_pause_time > 1:  # Debounce for 1 second
                    pyautogui.press('space')  # Play/Pause
                    play_pause_time = current_time
                play_pause_prev = True
            else:
                play_pause_prev = False

        # Perform scrolling action
        if scroll_direction == 1:
            pyautogui.scroll(3)  # Scroll Up
        elif scroll_direction == -1:
            pyautogui.scroll(-3)  # Scroll Down

    # Display the image
    cv2.imshow('Virtual Mouse', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()