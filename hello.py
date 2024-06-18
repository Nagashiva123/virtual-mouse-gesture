import cv2
import mediapipe as mp
import pyautogui
import numpy as np

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

            # Extract coordinates of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_x, index_finger_y = hand_landmarks_to_screen(index_finger_tip, image_width, image_height)

            # Map hand coordinates to screen coordinates
            screen_x = np.interp(index_finger_x, (0, image_width), (0, screen_width))
            screen_y = np.interp(index_finger_y, (0, image_height), (0, screen_height))

            # Move the mouse to the index finger tip position
            pyautogui.moveTo(screen_x, screen_y)

            # Check if the thumb and index finger are close to each other for a click gesture
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_x, thumb_y = hand_landmarks_to_screen(thumb_tip, image_width, image_height)
            distance = np.hypot(index_finger_x - thumb_x, index_finger_y - thumb_y)

            if distance < 30:
                pyautogui.click()

    # Display the image
    cv2.imshow('Virtual Mouse', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
