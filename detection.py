
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile

class PlayerDetector:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')  # Load pretrained model
        self.class_ids = [0]  # Person class in COCO
        
    def detect(self, frame):
        import os
        # Save frame to a temporary file to avoid numpy ambiguity in YOLO
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, frame)
            tmp_path = tmp.name

        results = self.model.predict(source=tmp_path, verbose=False)

        detections = []
        # Handle both single and list outputs
        if not isinstance(results, list):
            results = [results]
        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None and hasattr(result, '__getitem__'):
                try:
                    boxes = result[0].boxes
                except Exception:
                    continue
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) in self.class_ids and box.conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(box.conf)
                        })
        # Remove the temp file
        os.remove(tmp_path)
        return detections