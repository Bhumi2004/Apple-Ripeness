import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model for apple detection
model = YOLO('yolov10n.pt')  # YOLO weights file for apple detection

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get predictions from the model
    results = model(frame)

    apple_count = 0

    # Process the results
    for result in results:
        labels = result.names
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for i, cls in enumerate(classes):
            if labels[int(cls)] == 'apple':  # Assuming 'apple' is the label for apple in your model
                xmin, ymin, xmax, ymax = map(int, boxes[i])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                apple_count += 1

    # Display the apple count on the frame
    cv2.putText(frame, f'Apples: {apple_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with detection
    cv2.imshow('Apple Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
