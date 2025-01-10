from ultralytics import YOLO
import cv2

# Load your custom YOLO model
model = YOLO("my_custom_model.pt")  # Replace with your model path

#%%
# Initialize webcam
cap = cv2.VideoCapture(0)

# Loop through webcam frames
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Optional: Resize frame for faster inference
        resized_frame = cv2.resize(frame, (640, 640))

        # Convert frame from BGR (OpenCV format) to RGB (required by YOLO)
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model.predict(source=frame_rgb, save=False, show=False)

        # Ensure there are results before proceeding
        if results and len(results) > 0:
            # Extract annotated frame
            annotated_frame = results[0].plot()  # Generate the annotated image
        else:
            annotated_frame = resized_frame  # Display the raw frame if no results

        # Convert back to BGR for correct display in OpenCV
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Display the annotated frame
        cv2.imshow("YOLO Classification Monitor", annotated_frame_bgr)
        
        for prob, cls_idx in zip(results[0].probs.top5conf, results[0].probs.top5):
            class_name = results[0].names[cls_idx]  # Map index to class name
            print(f"Class: {class_name}, Probability: {prob:.2f}")
            
        # Break the loop on 'q' or 'ESC' key press
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # 27 is the ASCII code for ESC
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
