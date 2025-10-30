import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time # Added for a brief pause after clicking

# -----------------------------------------------------------------
# 1. INITIAL SETUP AND HELPER FUNCTIONS
# -----------------------------------------------------------------

def calculate_euclidean_distance(p1, p2, w, h):
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def count_fingers(landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers_up = 0

    if landmarks[tip_ids[0]].x > landmarks[tip_ids[0]-1].x:
        fingers_up += 1

    for id in range(1, 5):
        if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y:
            fingers_up += 1

    return fingers_up

# -----------------------------------------------------------------
# 2. MEDIAPIPE INITIALIZATION
# -----------------------------------------------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, # Only track one hand for simplicity
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize MediaPipe drawing utility
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

# -----------------------------------------------------------------
# 3. GLOBAL STATE VARIABLES
# -----------------------------------------------------------------

# Variables for 3-Finger Zoom Out (Kept from original code)
distance_history = []
MAX_HISTORY = 5      # Store distance for the last 5 frames
ZOOM_OUT_THRESHOLD = 15  # Pixel change threshold for 'zoom out' (expanding)
ZOOM_IN_THRESHOLD = -15 # Pixel change threshold for 'zoom in' (shrinking) - now negative
zoomed_flag = False

# -----------------------------------------------------------------
# 4. MAIN LOOP
# -----------------------------------------------------------------

class GestureDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.thumbsUpCooldown = 0
        self.bunnyEarsCooldown = 0
        self.zoomCooldown = 0

        self.thumbsUpCallback = None
        self.bunnyEarsCallback = None
        self.zoomOutCallback = None
        self.zoomInCallback = None

    def decrementCooldowns(self):
        self.thumbsUpCooldown = max(0, self.thumbsUpCooldown - 1)
        self.bunnyEarsCooldown = max(0, self.bunnyEarsCooldown - 1)
        self.zoomCooldown = max(0, self.zoomCooldown - 1)

    def onThumbsUp(self, callback):
        self.thumbsUpCallback = callback

    def onBunnyEars(self, callback):
        self.bunnyEarsCallback = callback

    def onZoomOut(self, callback):
        self.zoomOutCallback = callback

    def onZoomIn(self, callback):
        self.zoomInCallback = callback
        
    def checkThumbsUp(self, hand_landmarks, num_fingers):
        if self.thumbsUpCooldown > 0:
            return False

        if num_fingers != 1:
            return False

        thumb_up = hand_landmarks.landmark[4].y < hand_landmarks.landmark[2].y < hand_landmarks.landmark[1].y < hand_landmarks.landmark[0].y
        thumb_above_index = hand_landmarks.landmark[4].y < hand_landmarks.landmark[8].y
        # index_in = hand_landmarks.landmark[8].x > hand_landmarks.landmark[6].x
        # middle_in = hand_landmarks.landmark[12].x > hand_landmarks.landmark[10].x
        # ring_in = hand_landmarks.landmark[16].x > hand_landmarks.landmark[14].x
        # pinky_in = hand_landmarks.landmark[20].x > hand_landmarks.landmark[18].x
        hand_vertical = hand_landmarks.landmark[5].y < hand_landmarks.landmark[9].y < hand_landmarks.landmark[13].y < hand_landmarks.landmark[17].y

        return thumb_up and thumb_above_index and hand_vertical
        # return thumb_up and thumb_above_index and hand_vertical and index_in and middle_in and ring_in and pinky_in

    def checkZoomOut(self, hand_landmarks, num_fingers):
        return False
        if self.zoomCooldown > 0:
            return False

        if num_fingers != 3:
            return False

        index_tip = hand_landmarks.landmark[8]
        ring_tip = hand_landmarks.landmark[16]
        
        current_distance = calculate_euclidean_distance(index_tip, ring_tip, self.w, self.h)
        
        distance_history.append(current_distance)
        if len(distance_history) > MAX_HISTORY:
            distance_history.pop(0)

        if len(distance_history) < MAX_HISTORY:
            return False

        max_distance_in_window = max(distance_history)
        distance_decrease = max_distance_in_window - current_distance
        overall_decrease = distance_decrease > ZOOM_OUT_THRESHOLD 
        return overall_decrease 

    def checkZoomIn(self, hand_landmarks, num_fingers):
        return False
        if self.zoomCooldown > 0:
            return False

        if num_fingers != 3:
            return False

        index_tip = hand_landmarks.landmark[8]
        ring_tip = hand_landmarks.landmark[16]
        
        current_distance = calculate_euclidean_distance(index_tip, ring_tip, self.w, self.h)
        
        distance_history.append(current_distance)
        if len(distance_history) > MAX_HISTORY:
            distance_history.pop(0)

        if len(distance_history) < MAX_HISTORY:
            return False

        start_distance = distance_history[0]
        end_distance = distance_history[-1]
        distance_change = end_distance - start_distance
        distance_decreased1 = (distance_history[-1] - distance_history[-3]) < 0
        distance_decreased2 = (distance_history[-3] - distance_history[-5]) < 0
        return distance_change < -ZOOM_OUT_THRESHOLD and distance_decreased1 and distance_decreased2


    def checkBunnyEars(self, hand_landmarks, num_fingers):
        if self.bunnyEarsCooldown > 0:
            return False
        if num_fingers != 2:
            return False

        landmarks = hand_landmarks.landmark
        ring_down = landmarks[16].y > landmarks[14].y
        pinky_down = landmarks[20].y > landmarks[18].y
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        return index_up and middle_up and ring_down and pinky_down

                
    def start(self):
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                return

            self.decrementCooldowns()

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            self.h, self.w, self.c = image.shape

            # Check if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    num_fingers = count_fingers(hand_landmarks.landmark)

                    if self.checkThumbsUp(hand_landmarks, num_fingers):
                        self.thumbsUpCooldown = 30
                        if self.thumbsUpCallback:
                            self.thumbsUpCallback()

                    if self.checkZoomOut(hand_landmarks, num_fingers):
                        self.zoomCooldown = 30
                        if self.zoomOutCallback:
                            self.zoomOutCallback()
                    
                    if self.checkBunnyEars(hand_landmarks, num_fingers):
                        self.bunnyEarsCooldown = 30
                        if self.bunnyEarsCallback:
                            self.bunnyEarsCallback()

                    if self.checkZoomIn(hand_landmarks, num_fingers):
                        self.zoomCooldown = 30
                        if self.zoomInCallback:
                            self.zoomInCallback()

            cv2.imshow('Hand Gesture Recognizer', image)
            
            # Exit loop on 'q' press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                return
                

g = GestureDetector()
# g.onThumbsUp(lambda: pyautogui.doubleClick(440, 503))
# g.onBunnyEars(lambda: pyautogui.doubleClick(540, 747))
# g.onZoomOut(lambda: print("ZOOM OUT!"))
# g.onZoomIn(lambda: print("ZOOM IN!"))
g.start()


