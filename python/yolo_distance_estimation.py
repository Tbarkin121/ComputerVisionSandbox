# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:16:20 2025

@author: Plutonium
"""

import cv2

from ultralytics import solutions

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init distance-calculation obj
distance = solutions.DistanceCalculation(model="yolo11n.pt", show=True)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    im0 = distance.calculate(im0)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()