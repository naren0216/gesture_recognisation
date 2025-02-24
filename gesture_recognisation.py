import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module
mp_drawing = mp.solutions.drawing_utils

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Function to detect specific gestures
def is_fist(hand_landmarks):
    """Check if the hand is a fist (all fingers are curled in)"""
    for i in range(5, 21, 4):  # Check if the tips of all fingers are curled in
        if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i-4].y:
            return False
    return True

def is_victory(hand_landmarks):
    """Check if the hand is in a victory sign (index and middle finger raised)"""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    return index_tip.y < middle_tip.y

def is_thumb_up(hand_landmarks):
    """Check if thumb is pointing upwards"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    return thumb_tip.y < thumb_base.y

def is_thumb_down(hand_landmarks):
    """Check if thumb is pointing downwards"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    return thumb_tip.y > thumb_base.y

def is_open_hand(hand_landmarks):
    """Check if the hand is open (fingers spread wide)"""
    distances = []
    for i in range(0, 4):
        distances.append(np.linalg.norm(np.array([hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y]) - np.array([hand_landmarks.landmark[i + 1].x, hand_landmarks.landmark[i + 1].y])))
    return all(dist > 0.1 for dist in distances)

def is_pinch(hand_landmarks):
    """Check if the thumb and index finger are close together (pinch gesture)"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance < 0.05

def is_peace(hand_landmarks):
    """Check if index and middle fingers are extended (peace sign)"""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    return index_tip.y < middle_tip.y

def is_rock_on(hand_landmarks):
    """Check if the pinky and index fingers are extended (rock on gesture)"""
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return pinky_tip.y < index_tip.y

def is_pointing(hand_landmarks):
    """Check if the index finger is extended and others curled"""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    other_fingers = [hand_landmarks.landmark[i].y for i in [4, 8, 12, 16, 20]]
    return index_tip.y < min(other_fingers)

# Track hand movements
prev_positions = {}

while cap.isOpened():
    ret, frame = cap.read()

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hand landmarks
    results = hands.process(rgb_frame)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect specific gestures
            if is_fist(hand_landmarks):
                cv2.putText(frame, "Gesture: Fist", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif is_victory(hand_landmarks):
                cv2.putText(frame, "Gesture: Victory", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif is_thumb_up(hand_landmarks):
                cv2.putText(frame, "Gesture: Thumb Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            elif is_thumb_down(hand_landmarks):
                cv2.putText(frame, "Gesture: Thumb Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            elif is_open_hand(hand_landmarks):
                cv2.putText(frame, "Gesture: Open Hand", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            elif is_pinch(hand_landmarks):
                cv2.putText(frame, "Gesture: Pinch", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            elif is_peace(hand_landmarks):
                cv2.putText(frame, "Gesture: Peace Sign", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif is_rock_on(hand_landmarks):
                cv2.putText(frame, "Gesture: Rock On", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            elif is_pointing(hand_landmarks):
                cv2.putText(frame, "Gesture: Pointing", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image with detected gestures
    cv2.imshow("Gesture Recognition", frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()


'''OK Gesture: Thumb and index fingers forming a circle.
Victory Gesture: Index and middle fingers raised.
Fist Gesture: All fingers curled in.
Thumb Up: Thumb extended upwards.
Thumb Down: Thumb extended downwards.
Open Hand: All fingers spread wide.
Pinch Gesture: Thumb and index fingers coming together.
Peace Sign: Index and middle fingers raised in a "V".
Rock On: Pinky and index fingers raised, other fingers curled.
Pointing Gesture: Index finger extended, others curled.'''