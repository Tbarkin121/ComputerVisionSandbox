# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:59:57 2025

@author: Plutonium
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('../../../data/Human Action Recognition'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#%%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

#%%

# Load dataset paths
train_path = "../../../data/Human Action Recognition/train"
test_path = "../../../data/Human Action Recognition/test"
train_csv_path = "../../../data/Human Action Recognition/Training_set.csv"
test_csv_path = "../../../data/Human Action Recognition/Testing_set.csv"

# Load CSV files
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Presenting the Raw Dataset and Dataset Description
print("First few rows of the dataset:")
display(train_df.head())
print("\nDataset Description:")
print(f"Number of records: {train_df.shape[0]}")
print(f"Number of features: {train_df.shape[1]}")
print("\nData Types:")
print(train_df.dtypes)
print("\nMissing Values Count:")
print(train_df.isnull().sum())


#%%
print(f"Duplicate rows: {train_df.duplicated().sum()}")
train_df = train_df.drop_duplicates()
#%%
# Images from each class
fig, axes = plt.subplots(3, 5, figsize=(20, 10))
axes = axes.flatten()
for idx, class_name in enumerate(train_df['label'].unique()):
    class_images = train_df[train_df['label'] == class_name]['filename'].values
    img = plt.imread(os.path.join(train_path, class_images[0]))
    axes[idx].imshow(img)
    axes[idx].set_title(class_name)
    axes[idx].axis('off')
plt.tight_layout()
plt.show()

#%%
# Distribution of classes in training set
fig = px.histogram(train_df, x='label', title='Distribution of Classes in Training Set')
fig.show()

#%%
from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)


#%%
# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

#%% Organize the HAR data for YOLO Classification Training
import os
import shutil
from sklearn.model_selection import train_test_split

def organize_dataset_with_split(train_df, train_path, output_path, test_split=0.2):
    """
    Organize dataset into train/test directories for YOLO classification using an 80-20 split.

    Args:
        train_df (pd.DataFrame): DataFrame containing filenames and labels for training.
        train_path (str): Path to the directory containing training images.
        output_path (str): Path to output the organized dataset.
        test_split (float): Proportion of data to use for testing.
    """
    os.makedirs(output_path, exist_ok=True)
    train_output_path = os.path.join(output_path, 'train')
    test_output_path = os.path.join(output_path, 'test')
    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(test_output_path, exist_ok=True)

    # Perform an 80-20 split for each label
    for label in train_df['label'].unique():
        label_data = train_df[train_df['label'] == label]
        train_data, test_data = train_test_split(label_data, test_size=test_split, random_state=42)

        # Function to copy images into the correct folder structure
        def copy_images(df, source_path, dest_path):
            for _, row in df.iterrows():
                filename = row['filename']
                src_file = os.path.join(source_path, filename)
                dest_dir = os.path.join(dest_path, label)

                # Create the label directory if it doesn't exist
                os.makedirs(dest_dir, exist_ok=True)

                # Copy the image
                if os.path.exists(src_file):
                    shutil.copy(src_file, os.path.join(dest_dir, filename))
                else:
                    print(f"File not found: {src_file}")

        # Organize training and testing datasets
        print(f"Organizing {label} training data...")
        copy_images(train_data, train_path, train_output_path)

        print(f"Organizing {label} testing data...")
        copy_images(test_data, train_path, test_output_path)

    print(f"Dataset organized successfully at {output_path}")

# Example usage
train_path = "../../../data/Human Action Recognition/train"
output_path = "har_organized_dataset"

organize_dataset_with_split(train_df, train_path, output_path, test_split=0.2)

#%%
