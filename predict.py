# predict.py
import cv2
import mediapipe as mp
import json
import os

mp_face = mp.solutions.face_detection
cap = cv2.VideoCapture(1)

MODEL_DIR = "models"
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(MODEL_DIR, "lbph.yml"))
with open(os.path.join(MODEL_DIR, "label_map.json"), "r") as f:
    label_map = json.load(f)
inv_map = {v:k for k,v in label_map.items()}

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.flip(frame, 1)
        h, w = img.shape[:2]
        results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w); y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w); y2 = y1 + int(bbox.height * h)
                pad = 10
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(w, x2+pad), min(h, y2+pad)
                face = img[y1:y2, x1:x2]
                if face.size == 0: continue
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (200,200))
                label_id, conf = recognizer.predict(face_resized)
                name = inv_map.get(label_id, "unknown")
                txt = f"{name} ({conf:.1f})"
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Predict", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Press any key in the OpenCV window to exit")
cv2.waitKey(0)
cv2.destroyAllWindows()