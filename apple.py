import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov10n.pt')  # YOLOv10 weights file

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    ripe_count = 0
    overripe_count = 0
    unripe_count = 0

    for result in results:
        labels = result.names
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for i, cls in enumerate(classes):
            if labels[int(cls)] == 'apple':  
                xmin, ymin, xmax, ymax = map(int, boxes[i])

                apple_region = frame[ymin:ymax, xmin:xmax]

                hsv = cv2.cvtColor(apple_region, cv2.COLOR_BGR2HSV)

                lower_ripe = np.array([10, 100, 100])
                upper_ripe = np.array([25, 255, 255])
                lower_overripe = np.array([0, 50, 50])
                upper_overripe = np.array([10, 255, 255])
                lower_unripe = np.array([35, 100, 100])
                upper_unripe = np.array([85, 255, 255])

                ripe_mask = cv2.inRange(hsv, lower_ripe, upper_ripe)
                overripe_mask = cv2.inRange(hsv, lower_overripe, upper_overripe)
                unripe_mask = cv2.inRange(hsv, lower_unripe, upper_unripe)

                ripe_ratio = cv2.countNonZero(ripe_mask) / (apple_region.shape[0] * apple_region.shape[1])
                overripe_ratio = cv2.countNonZero(overripe_mask) / (apple_region.shape[0] * apple_region.shape[1])
                unripe_ratio = cv2.countNonZero(unripe_mask) / (apple_region.shape[0] * apple_region.shape[1])

                if ripe_ratio > overripe_ratio and ripe_ratio > unripe_ratio:
                    ripe_count += 1
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                elif overripe_ratio > ripe_ratio and overripe_ratio > unripe_ratio:
                    overripe_count += 1
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                else:
                    unripe_count += 1
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.putText(frame, f'Ripe: {ripe_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Overripe: {overripe_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Unripe: {unripe_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Apple Classification', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()     

