import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time # Added for a brief pause after clicking

# -----------------------------------------------------------------
# 1. INITIAL SETUP AND HELPER FUNCTIONS
# -----------------------------------------------------------------

def calculate_euclidean_distance(p1, p2, w, h):
    """Calculates the Euclidean distance in pixel space between two normalized landmarks."""
    # Convert normalized coordinates (0 to 1) to pixel coordinates
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    
    # Calculate Euclidean distance
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def count_fingers(landmarks):
    """Counts how many non-thumb fingers are up based on Y-coordinate comparison."""
    # Tip IDs for all 5 fingers (0=wrist, 4=thumb tip, 8=index tip, etc.)
    tip_ids = [4, 8, 12, 16, 20]
    fingers_up = 0

    # 1. Thumb (Tip 4) - Check X-axis for thumb extended right (simple horizontal check)
    # This check is basic and can be improved, but is kept for consistency with original code structure
    if landmarks[tip_ids[0]].x > landmarks[tip_ids[0]-1].x:
        fingers_up += 1

    # 2-5. Other Fingers (Tips 8, 12, 16, 20)
    for id in range(1, 5):
        # Check if the fingertip's Y-coordinate is higher (smaller value) than the Y-coordinate of the first knuckle (extended)
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

# Flag to prevent repeated clicks while holding the Thumbs-Up pose
clicked_flag = False

# Variables for 3-Finger Zoom Out (Kept from original code)
distance_history = []
MAX_HISTORY = 5      # Store distance for the last 5 frames
ZOOM_OUT_THRESHOLD = 15  # Pixel change threshold for 'zoom out' (expanding)
ZOOM_IN_THRESHOLD = -15 # Pixel change threshold for 'zoom in' (shrinking) - now negative
zoomed_flag = False

# -----------------------------------------------------------------
# 4. MAIN LOOP
# -----------------------------------------------------------------

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a natural selfie-view and convert BGR to RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hand landmarks
    results = hands.process(image_rgb)
    
    # Get the image dimensions for pixel calculations
    h, w, c = image.shape 

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Use the finger count logic
            num_fingers = count_fingers(hand_landmarks.landmark)

            # --- THUMBS-UP GESTURE LOGIC (Replaces Wag) ---
            
            # Check if all four non-thumb fingers are down
            # The count_fingers logic will return 1 if only the thumb is up
            if num_fingers == 1:
                # To make the Thumbs-Up check more robust:
                # Check that the Index Finger tip (8) is BELOW its knuckle (6)
                index_down = hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y
                
                # Check that the thumb tip (4) is above its knuckle (2)
                thumb_up = hand_landmarks.landmark[4].y < hand_landmarks.landmark[2].y

                thumb_above_index = hand_landmarks.landmark[4].y < hand_landmarks.landmark[8].y

                hand_vertical = hand_landmarks.landmark[5].y < hand_landmarks.landmark[9].y < hand_landmarks.landmark[13].y < hand_landmarks.landmark[17].y
                print(hand_vertical)

                if index_down and thumb_up and thumb_above_index and hand_vertical:
                    cv2.putText(image, 'GESTURE: THUMBS-UP! 👍', (10, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 3, cv2.LINE_AA)
                    
                    if not clicked_flag:
                        # Trigger the click at (530, 530)
                        pyautogui.click(530, 530, duration=0)
                        clicked_flag = True # Set the flag to prevent rapid-fire clicks
                        print("THUMBS-UP detected. Click triggered at (530, 530).")
                        # Add a small delay to prevent immediate re-triggering
                        time.sleep(0.5) 
                else:
                    # Pose is 1 finger up, but not a reliable Thumbs-Up (e.g., just the Index)
                    clicked_flag = False
            else:
                # Any other number of fingers up
                clicked_flag = False

            # --- BUNNY EARS GESTURE LOGIC (NEW) ---
            
            # Check if exactly 2 fingers are counted as up
            landmarks = hand_landmarks.landmark
            if num_fingers == 2:
                
                
                # Check 2: Ring finger (Tip 16) is definitely DOWN (lower than its knuckle 14)
                ring_down = landmarks[16].y > landmarks[14].y
                
                # Check 3: Pinky finger (Tip 20) is definitely DOWN (lower than its knuckle 18)
                pinky_down = landmarks[20].y > landmarks[18].y
                
                # Check 4: Index finger (Tip 8) is UP
                index_up = landmarks[8].y < landmarks[6].y
                
                # Check 5: Middle finger (Tip 12) is UP
                middle_up = landmarks[12].y < landmarks[10].y

                # Final Bunny Ears (V-Sign/Peace) check
                if index_up and middle_up and ring_down and pinky_down:
                    cv2.putText(image, 'GESTURE: BUNNY EARS! ✌️', (10, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 3, cv2.LINE_AA)
                    
                    if not clicked_flag:
                        # Trigger the click at (530, 530)
                        clicked_flag = True # Set the flag to prevent rapid-fire clicks
                        print("BUNNY EARS detected. Click triggered at (530, 530).")
                        pyautogui.doubleClick(1000, 715, duration=0)

                        # Add a small delay to prevent immediate re-triggering
                        time.sleep(0.5) 
                else:
                    # 2 fingers up, but not Index/Middle (e.g., Thumb and Index)
                    clicked_flag = False
            else:
                # Any other number of fingers up
                clicked_flag = False

            # --- 3-FINGER ZOOM GESTURE LOGIC (Expanding and Shrinking) ---

            if num_fingers == 3:
                # Measure distance between Index Tip (8) and Ring Tip (16)
                index_tip = landmarks[8]
                ring_tip = landmarks[16]
                
                current_distance = calculate_euclidean_distance(index_tip, ring_tip, w, h)
                
                distance_history.append(current_distance)
                if len(distance_history) > MAX_HISTORY:
                    distance_history.pop(0) 

                if len(distance_history) == MAX_HISTORY:
                    start_distance = distance_history[0]
                    end_distance = distance_history[-1]
                    
                    # Calculate the change in distance
                    distance_change = end_distance - start_distance

                    # --- ZOOM OUT (Expanding) ---
                    if distance_change > ZOOM_OUT_THRESHOLD and not zoomed_flag:
                        cv2.putText(image, 'GESTURE: ZOOM OUT! 🤏', (10, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                        pyautogui.press('esc') # Or use your specific zoom out command
                        print("ZOOM OUT detected!")
                        distance_history = [] # Reset history to prevent re-trigger
                        zoomed_flag = True # Set flag
                    
                    # --- ZOOM IN (Shrinking) ---
                    elif distance_change < ZOOM_IN_THRESHOLD and not zoomed_flag: # Check for negative change
                        cv2.putText(image, 'GESTURE: ZOOM IN! 🤏', (10, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA) # Different color for clarity
                        pyautogui.click(1000, 770, duration=0) 
                        print("ZOOM IN detected!")
                        distance_history = [] # Reset history to prevent re-trigger
                        zoomed_flag = True # Set flag
                    else:
                        zoomed_flag = False
                        
            else:
                # If the pose is broken or not 3 fingers, clear the zoom history and reset flag
                distance_history = []
                zoomed_flag = False
    # Convert the processed image back to BGR and display it
    cv2.imshow('Hand Gesture Recognizer', image)
    
    # Exit loop on 'q' press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the camera and destroy windows
