# capture.py
import cv2
import mediapipe as mp
import os
import json

mp_face = mp.solutions.face_detection
cap = cv2.VideoCapture(1)

# Debug: Check if camera opened successfully
if not cap.isOpened():
    print("ERROR: Could not open camera!")
    exit(1)
else:
    print("Camera opened successfully")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
label = input("Label (person name, e.g. alice): ").strip().lower()
label_dir = os.path.join(DATA_DIR, label)
os.makedirs(label_dir, exist_ok=True)

count = len(os.listdir(label_dir))
target = 50  # number of face images to capture

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
    print(f"Starting capture for label: {label}")
    print(f"Target: {target} images")
    print("Press 'q' to quit early")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            print(f"ERROR: Failed to read frame {frame_count}")
            break
            
        if frame_count % 30 == 0:  # Print every 30 frames (~1 second)
            print(f"Processing frame {frame_count}, captured {count}/{target} faces")
            
        img = cv2.flip(frame, 1)
        h, w = img.shape[:2]
        results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w)
                y2 = y1 + int(bbox.height * h)
                # pad and clamp coords
                pad = 10
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                face = img[y1:y2, x1:x2]
                if face.size > 0:
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    fname = os.path.join(label_dir, f"{label}_{count:03d}.png")
                    cv2.imwrite(fname, face_gray)
                    print(f"Saved: {fname}")
                    count += 1
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f"{label}: {count}/{target}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Capture", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or count >= target:
            break
        

cap.release()
cv2.destroyAllWindows()
print("Captured:", count)
