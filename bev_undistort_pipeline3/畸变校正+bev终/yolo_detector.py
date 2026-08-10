"""Small, testable YOLO adapter."""

from __future__ import annotations
import numpy as np

VRU_LABELS = {"person", "bicycle", "motorcycle", "bike", "cyclist", "pedestrian", "rider"}

#执行六路YOLO检测
def detect_images(model, images: dict[str, np.ndarray], imgsz=640, device="cpu", confidence=0.25):
    output = {}
    for camera, image in images.items():
        result = model.predict(source=image, device=device, imgsz=imgsz, conf=confidence, verbose=False)[0]
        names = result.names
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for box, score, class_id in zip(boxes, scores, classes):
                label = names[int(class_id)]
                if label.lower() in VRU_LABELS:
                    detections.append({"class": int(class_id), "label": label, "score": float(score), "bbox": box.astype(float).tolist()})
        output[camera] = detections
    return output
