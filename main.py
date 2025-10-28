import cv2
import mediapipe as mp
import numpy as np

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
# 3. GLOBAL STATE VARIABLES FOR DYNAMIC GESTURES
# -----------------------------------------------------------------

# Variables for 3-Finger Zoom Out
distance_history = []
MAX_HISTORY = 5      # Store distance for the last 5 frames
ZOOM_THRESHOLD = 30  # Pixel change threshold for 'zoom out'

# Variables for Pointer Finger Wag
wag_x_history = [] 
WAG_HISTORY_LENGTH = 5              # Analyze the last 5 frames for quick movement
WAG_DISTANCE_THRESHOLD = 0.03       # Normalized distance (3% of screen width) for movement
current_wag_state = 0               # 0: Idle, 1: Moved one direction, 2: Completed 1 wag
wag_count = 0 


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

            # Display the basic result
            cv2.putText(image, f'Fingers Up: {num_fingers}', (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # --- POINTER FINGER WAG GESTURE LOGIC ---
            
            # 1. Get the Index Finger Tip (Landmark 8) X-coordinate
            index_tip_x = hand_landmarks.landmark[8].x
            
            # Check if the pose is right (ONLY index finger up)
            # The second condition checks if index tip is higher than the knuckle below it
            if num_fingers == 1 and hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y: 
                
                wag_x_history.append(index_tip_x)
                if len(wag_x_history) > WAG_HISTORY_LENGTH:
                    wag_x_history.pop(0)

                # Need enough history to compare start and end movement
                if len(wag_x_history) >= WAG_HISTORY_LENGTH:
                    start_x = wag_x_history[0]
                    end_x = wag_x_history[-1]
                    horizontal_change = end_x - start_x

                    # State 0: Idle / Waiting for first move
                    if current_wag_state == 0:
                        if abs(horizontal_change) > WAG_DISTANCE_THRESHOLD:
                            # Move detected, transition to State 1
                            current_wag_state = 1
                            wag_x_history = wag_x_history[-1:] # Reset history to start monitoring the return move
                    
                    # State 1: Moved one direction, waiting for the return move (first wag)
                    elif current_wag_state == 1:
                        # Check for movement back (change is opposite of initial movement)
                        if abs(horizontal_change) > WAG_DISTANCE_THRESHOLD and np.sign(horizontal_change) != np.sign(wag_x_history[0] - wag_x_history[1]):
                            wag_count += 1
                            current_wag_state = 2 # Completed 1 wag
                            wag_x_history = wag_x_history[-1:] # Reset history for the second wag
                    
                    # State 2: Completed 1 wag, waiting for the final move (second wag)
                    elif current_wag_state == 2:
                        if abs(horizontal_change) > WAG_DISTANCE_THRESHOLD:
                            wag_count += 1
                            current_wag_state = 0 # Completed 2 wags, reset state
                            wag_x_history = []

                # Display the wag count
                cv2.putText(image, f'Wags Detected: {wag_count}', (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2, cv2.LINE_AA)
                            
                # Final check for the completed gesture
                if wag_count >= 2:
                    cv2.putText(image, 'GESTURE: DOUBLE WAG!', (10, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 3, cv2.LINE_AA)
                    # Reset the wag state after detection
                    wag_count = 0
                    current_wag_state = 0
                    
            else:
                # If the hand pose is broken, reset all wag tracking
                wag_count = 0
                current_wag_state = 0
                wag_x_history = []

            # --- 3-FINGER ZOOM OUT GESTURE LOGIC ---

            # Check for the 3-Finger Static Pose
            if num_fingers == 3:
                
                # Measure distance between Index Tip (8) and Ring Tip (16)
                index_tip = hand_landmarks.landmark[8]
                ring_tip = hand_landmarks.landmark[16]
                
                current_distance = calculate_euclidean_distance(index_tip, ring_tip, w, h)
                
                # Update distance history
                distance_history.append(current_distance)
                if len(distance_history) > MAX_HISTORY:
                    distance_history.pop(0) # Remove oldest distance

                # Check for Zoom-Out (Distance Increasing)
                if len(distance_history) == MAX_HISTORY:
                    start_distance = distance_history[0]
                    end_distance = distance_history[-1]
                    
                    # Check if the change is positive (zoom out) and exceeds the pixel threshold
                    if (end_distance - start_distance) > ZOOM_THRESHOLD:
                        cv2.putText(image, 'GESTURE: ZOOM OUT!', (10, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                        
                        # Clear the history to prevent continuous triggering
                        distance_history = []
                        
            else:
                # If the pose is broken, clear the zoom history
                distance_history = []


    # Convert the processed image back to BGR and display it
    cv2.imshow('Hand Gesture Recognizer', image)
    
    # Exit loop on 'q' press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the camera and destroy windows
cap.release()
cv2.destroyAllWindows()
