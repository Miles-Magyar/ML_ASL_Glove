#THIS WAS AI ASSISTED CODE


import os
import sys
import cv2
import numpy as np
import ai_edge_litert.interpreter as tflite
import time

LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
IMAGES_PER_LETTER = 100
DATASET_DIR = "dataset"
INVERT_IMAGES = False

CUT_TOP = 0.3      
CUT_BOTTOM = 0.0   
CUT_SIDES = 0.15   
MASK_INFLATION = 60    
CONFIDENCE_THRESH = 0.5

os.makedirs(DATASET_DIR, exist_ok=True)
for label in LABELS:
    os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)

def sigmoid(x): 
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

cwd = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
models = {}
for filename, key in [("detector.tflite", "det"), ("landmarks.tflite", "lm")]:
    path = os.path.join(cwd, filename)
    if not os.path.exists(path):
        print(f"MISSING: {filename}")
        sys.exit()
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()
    models[key] = {
        "model": interp,
        "in": interp.get_input_details()[0]['index'],
        "outs": interp.get_output_details()
    }

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press SPACE to start capturing for each letter.")
print("Press ESC to exit early.")

for current_letter in LABELS:
    capturing = False
    saved_count = 0
    
    while saved_count < IMAGES_PER_LETTER:
        ok, raw_frame = cap.read()
        if not ok: 
            continue
            
        raw_frame = cv2.flip(raw_frame, 1)
        
        h_raw, w_raw, _ = raw_frame.shape
        y1 = int(h_raw * CUT_TOP)
        y2 = int(h_raw * (1.0 - CUT_BOTTOM))
        x1 = int(w_raw * CUT_SIDES)
        x2 = int(w_raw * (1.0 - CUT_SIDES))
        
        frame = raw_frame[y1:y2, x1:x2].copy()
        frame = cv2.resize(frame, (640, 480))
        h, w, _ = frame.shape
        clean_frame = frame.copy()
        
        if not capturing:
            cv2.putText(frame, f"Letter: {current_letter} - Press SPACE to start", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Capturing {current_letter}... {saved_count}/{IMAGES_PER_LETTER}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.putText(frame, "Move hand slightly to create variance!", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        img_in = cv2.resize(frame, (256, 256))
        img_rgb = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)
        
        models["det"]["model"].set_tensor(models["det"]["in"], input_data)
        models["det"]["model"].invoke()
        
        out0 = models["det"]["model"].get_tensor(models["det"]["outs"][0]['index'])
        out1 = models["det"]["model"].get_tensor(models["det"]["outs"][1]['index'])
        scores = out0.flatten() if out0.shape[-1] < out1.shape[-1] else out1.flatten()
        
        final_in = None
        
        if np.max(sigmoid(scores)) > CONFIDENCE_THRESH:
            models["lm"]["model"].set_tensor(models["lm"]["in"], input_data)
            models["lm"]["model"].invoke()
            
            lm_raw = None
            for out in models["lm"]["outs"]:
                t = models["lm"]["model"].get_tensor(out['index']).flatten()
                if len(t) >= 63: 
                    lm_raw = t
                    break
                
            if lm_raw is not None:
                max_val = np.max(lm_raw)
                points_list = []
                
                for i in range(0, 63, 3):
                    vx, vy = lm_raw[i], lm_raw[i+1]
                    if max_val <= 1.0: 
                        cx, cy = int(vx * w), int(vy * h)
                    else: 
                        cx, cy = int((vx/256)*w), int((vy/256)*h)
                    points_list.append([cx, cy])
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)

                if len(points_list) > 0:
                    mask = np.zeros((h, w), dtype=np.uint8)
                    hull = cv2.convexHull(np.array(points_list, dtype=np.int32))
                    cv2.fillConvexPoly(mask, hull, 255)
                    kernel = np.ones((MASK_INFLATION, MASK_INFLATION), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                    
                    white_bg = np.ones_like(clean_frame) * 255
                    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    final_comp = np.where(mask_3ch > 0, clean_frame, white_bg)
                    
                    hx, hy, hw, hh = cv2.boundingRect(np.array(points_list, dtype=np.int32))
                    dynamic_size = max(hw, hh) + 40
                    center_x, center_y = hx + hw // 2, hy + hh // 2
                    
                    dx1 = max(0, center_x - dynamic_size // 2)
                    dy1 = max(0, center_y - dynamic_size // 2)
                    dx2 = min(w, center_x + dynamic_size // 2)
                    dy2 = min(h, center_y + dynamic_size // 2)
                    
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                    roi = final_comp[dy1:dy2, dx1:dx2]
                    
                    if roi.size > 0:
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        final_in = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
                        
                        if INVERT_IMAGES:
                            final_in = cv2.bitwise_not(final_in)
                        
                        view_large = cv2.resize(final_in, (140, 140), interpolation=cv2.INTER_NEAREST)
                        frame[0:140, w-140:w] = cv2.cvtColor(view_large, cv2.COLOR_GRAY2BGR)
        
        cv2.imshow("Data Collection Pipeline", frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == 27: 
            print("Exiting.")
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

print("\nData collection complete!")
cap.release()
cv2.destroyAllWindows()