#THIS WAS AI ASSISTED CODE


import os
import sys
import cv2
import numpy as np
import time

LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
IMAGES_PER_LETTER = 100
DATASET_DIR = "glove_dataset"
INVERT_IMAGES = False

H_MIN, S_MIN, V_MIN = 37, 33, 102
H_MAX, S_MAX, V_MAX = 82, 205, 255

os.makedirs(DATASET_DIR, exist_ok=True)
for label in LABELS:
    os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)

def nothing(x):
    pass

cv2.namedWindow("Glove Data Collection")
cv2.createTrackbar("H Min", "Glove Data Collection", H_MIN, 179, nothing)
cv2.createTrackbar("H Max", "Glove Data Collection", H_MAX, 179, nothing)
cv2.createTrackbar("S Min", "Glove Data Collection", S_MIN, 255, nothing)
cv2.createTrackbar("S Max", "Glove Data Collection", S_MAX, 255, nothing)
cv2.createTrackbar("V Min", "Glove Data Collection", V_MIN, 255, nothing)
cv2.createTrackbar("V Max", "Glove Data Collection", V_MAX, 255, nothing)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press SPACE to start capturing for each letter.")
print("Press ESC to exit.")

for current_letter in LABELS:
    capturing = False
    saved_count = 0
    
    while saved_count < IMAGES_PER_LETTER:
        ok, raw_frame = cap.read()
        if not ok: 
            continue
            
        frame = cv2.flip(raw_frame, 1)
        h, w, _ = frame.shape
        
        # Center target box dimensions
        box_size = 280
        cx, cy = w // 2, h // 2
        fx1, fy1 = max(0, cx - box_size // 2), max(0, cy - box_size // 2)
        fx2, fy2 = min(w, cx + box_size // 2), min(h, cy + box_size // 2)

        # Get HSV values from trackbars
        h_min = cv2.getTrackbarPos("H Min", "Glove Data Collection")
        h_max = cv2.getTrackbarPos("H Max", "Glove Data Collection")
        s_min = cv2.getTrackbarPos("S Min", "Glove Data Collection")
        s_max = cv2.getTrackbarPos("S Max", "Glove Data Collection")
        v_min = cv2.getTrackbarPos("V Min", "Glove Data Collection")
        v_max = cv2.getTrackbarPos("V Max", "Glove Data Collection")

        # Crop ROI and extract Glove Mask
        roi_color = frame[fy1:fy2, fx1:fx2]
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        roi_mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Draw green target box on frame
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)

        final_in = None
        if cv2.countNonZero(roi_mask) > 500:
            final_in = cv2.resize(roi_mask, (28, 28), interpolation=cv2.INTER_AREA)
            if INVERT_IMAGES:
                final_in = cv2.bitwise_not(final_in)
            
            # Display miniature preview in top-right corner
            view_large = cv2.resize(final_in, (140, 140), interpolation=cv2.INTER_NEAREST)
            frame[0:140, w-140:w] = cv2.cvtColor(view_large, cv2.COLOR_GRAY2BGR)

        # UI Overlay Text
        if not capturing:
            cv2.putText(frame, f"Letter: {current_letter} - Press SPACE to start", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Capturing {current_letter}... {saved_count}/{IMAGES_PER_LETTER}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        cv2.imshow("Glove Data Collection", frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == 27: 
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
            
        if not capturing:
            if key == ord(' '):
                capturing = True
                print(f"Capturing letter {current_letter}...")
        else:
            if final_in is not None:
                file_path = os.path.join(DATASET_DIR, current_letter, f"{saved_count:03d}.png")
                cv2.imwrite(file_path, final_in)
                saved_count += 1
                time.sleep(0.05) 

print("\nGlove Data Collection Complete!")
cap.release()
cv2.destroyAllWindows()