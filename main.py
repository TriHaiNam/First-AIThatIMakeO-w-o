import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolo11n.pt')

# Start the video capture from USB camera
cap = cv2.VideoCapture(0)  # Use 0 or 1 depending on your USB camera index
cap.set(cv2.CAP_PROP_FPS, 60)  # Set the frame rate to 60 fps

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Filter results to only include "person"
    person_detected = False
    people_count = 0
    for result in results:
        for box in result.boxes:
            if model.names[int(box.cls)] == 'person':
                person_detected = True
                people_count += 1
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Extract values from tensor and convert to list
                label = model.names[int(box.cls)]
                confidence = box.conf[0] * 100  # Get confidence score and convert to percentage
                label_with_confidence = f"{label} {confidence:.2f}%"

                # Calculate color based on confidence
                color = (0, int(255 * (confidence / 100)), int(255 * (1 - confidence / 100)))

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label_with_confidence, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame only if a person is detected
    if person_detected:
        # Display the count of people
        cv2.putText(frame, f"People count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()