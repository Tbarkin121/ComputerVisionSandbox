from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="har_organized_dataset", epochs=100, imgsz=640)



#%%
# success = model.export()
# Load the best weights after training
best_model = YOLO("runs/classify/train5/weights/best.pt")

# Save the model to a custom path
best_model.save("my_custom_model.pt")
    
#%%
# Load the saved model
saved_model = YOLO("my_custom_model.pt")

# Run inference
results = saved_model.predict(source="har_organized_dataset/train/laughing/Image_18.jpg")

#%%

