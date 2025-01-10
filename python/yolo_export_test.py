# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:19:05 2025

@author: Plutonium
"""
from ultralytics import YOLO

#%%
# load an official model
model11 = YOLO("yolo11n.pt")
# model11_pose = YOLO("yolo11n-pose.pt")  
# model10 = YOLO("yolov10n.pt")
model9 = YOLO("yolov9c.pt")
#%%
# Export the model
model11.export(format="onnx", simplify=True, opset=15)
# model11_pose.export(format="onnx", simplify=False)
# model10.export(format="onnx", simplify=False)
model9.export(format="onnx", simplify=False, opset=15)

# model11.export(format="tflite") #Need to downgrade to pytho 2.11 or earlier...
# model11_pose.export(format="tflite") #Need to downgrade to pytho 2.11 or earlier...

# model.export(format="onnx", int8=True)

#%%
model11.info()
# #%%
# # Train the model on the COCO8 example dataset for 100 epochs
# results = model11.train(data="coco8.yaml", epochs=100, imgsz=640)

# # Run inference with the YOLO11n model on the 'bus.jpg' image
# results = model11("path/to/bus.jpg")


# #%%

# import onnxruntime as ort
# import numpy as np
# import torch

# session = ort.InferenceSession("yolo11n.onnx")
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name

# # Prepare dummy input
# dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

# # Execute the model
# outputs = session.run([output_name], {input_name: dummy_input})


# outputs_tensor = torch.tensor(outputs)
# print(outputs_tensor.shape)