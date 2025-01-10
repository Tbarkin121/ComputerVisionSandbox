#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 00:58:35 2025

@author: tylerbarkin
"""
from ultralytics import YOLO
import cv2
import pygame
import time
import threading

#%%

# Confidence threshold to filter keypoints
CONF_THRESHOLD = 0.5

# Keypoint labels (based on COCO format or model-specific structure)
keypoint_labels = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]



# Initialize models
model_base = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model_pose = YOLO("yolo11n-pose.pt")
model_seg = YOLO("yolo11n-seg.pt")
model_class = YOLO("yolo11n-cls.pt")

# models_to_run = [model_base, model_pose]
models_to_run = [model_base]

#%%
def play_chime():
    pygame.mixer.init()
    pygame.mixer.music.load("chime.wav")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait for the sound to finish
        continue
    
audio_lock = threading.Lock()
def play_chime2():
    """Plays the chime sound asynchronously."""
    def play_sound():
        with audio_lock:  # Acquire the lock before playing sound
            pygame.mixer.init()
            pygame.mixer.music.load("chime2.wav")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue  # Keep the thread alive until the sound finishes

    # Start a new thread to play the sound
    if not audio_lock.locked():
        threading.Thread(target=play_sound, daemon=True).start()

def process_frame(frame):
    # Resize the frame for better performance
    # resized_frame = cv2.resize(frame, (640, 640))
    
    # Convert frame from BGR (OpenCV format) to RGB (required by YOLOv8)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    # Perform inference on the frame
    results_list = []
    annotated_frame = frame.copy()  
    
    for model in models_to_run:
        
        results = model.predict(frame_rgb, save=False, show=False)
        
        results_list.append(results)
        
        # Plot the annotations if applicable
        # if hasattr(result[0], "plot"):  # Check if the model has a 'plot' method
        annotated_frame = results[0].plot(img=annotated_frame)

        
    return results_list, annotated_frame

def print_detected_keypoints(results_list):
    for results in results_list:
        for result in results:
            keypoints = result.keypoints
            if keypoints is not None:
                confidences = keypoints.conf  # Get confidence for all keypoints
                print("Detected Keypoints:")
                for i, conf in enumerate(confidences[0]):  # confidences[0] for the first detected person
                    if conf >= CONF_THRESHOLD:  # Check if confidence is above the threshold
                        label = keypoint_labels[i] if i < len(keypoint_labels) else f"Point {i}"
                        print(f"  {label}: confidence={conf:.2f}")


def check_and_trigger_chime(results_list):
    """Check for cellphone detection and play chime if detected with high confidence."""
    for results in results_list:
        for result in results:
            if hasattr(result, "boxes") and result.boxes is not None:
                # Access the boxes object
                boxes = result.boxes
                
                # cls contains the detected object classes
                detected_classes = boxes.cls.cpu().numpy()  # Convert to NumPy for easy processing
                confidences = boxes.conf.cpu().numpy()  # Confidence scores
                
                # Check for cellphone (class 67) with high confidence
                for cls, conf in zip(detected_classes, confidences):
                    if int(cls) == 67 and conf > 0.8:  # 67 is the class ID for cellphone
                        print(f"Cellphone detected with confidence: {conf:.2f}")
                        play_chime2()
                        return  # Play chime only once per frame
    
play_chime()

# Initialize webcam
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        

        # Perform classification
        results_list, annotated_frame = process_frame(frame)
        
        # Display the annotated frame
        cv2.imshow("YOLOv11 Pose Monitor", annotated_frame)

        # print_detected_keypoints(results_list)
        check_and_trigger_chime(results_list)


        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    

    
#%%

    