from ultralytics import YOLO
import cv2

# Load the YOLO pose model
model = YOLO("yolo11n-pose.pt")  # load an official model

# Export the model
# model.export(format="onnx")

# Keypoint labels (based on COCO format or model-specific structure)
keypoint_labels = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# Confidence threshold to filter keypoints
CONF_THRESHOLD = 0.5
# Initialize webcam
cap = cv2.VideoCapture(0)

# Loop through frames
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR (OpenCV format) to RGB (required by YOLOv8)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model.predict(source=frame_rgb, save=False, show=False)
        
        # Extract annotated frame
        annotated_frame = results[0].plot()  # 'plot()' generates the annotated image

        # Display the annotated frame
        cv2.imshow("YOLOv8 Pose Monitor", annotated_frame)

        
        # Process keypoints and print visible points
        for result in results:
            keypoints = result.keypoints  # Access keypoints object
            if keypoints is not None:
                confidences = keypoints.conf  # Get confidence for all keypoints
                print("Detected Keypoints:")
                for i, conf in enumerate(confidences[0]):  # confidences[0] for the first detected person
                    if conf >= CONF_THRESHOLD:  # Check if confidence is above the threshold
                        label = keypoint_labels[i] if i < len(keypoint_labels) else f"Point {i}"
                        print(f"  {label}: confidence={conf:.2f}")


        
        # Break the loop on 'q' or 'ESC' key press
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # 27 is the ASCII code for ESC
            break
finally:
    cap.release()
    cv2.destroyAllWindows()