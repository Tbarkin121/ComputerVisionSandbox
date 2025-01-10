# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:04:25 2025

@author: Plutonium
"""

import os

def explore_video_folder(base_path):
    """
    Explore the SPHAR video folder structure and count videos per label.

    Args:
        base_path (str): Path to the folder containing labeled subdirectories.

    Returns:
        dict: A dictionary with label names as keys and video counts as values.
    """
    label_counts = {}

    # Walk through the base directory
    for label_folder in os.listdir(base_path):
        label_path = os.path.join(base_path, label_folder)

        # Check if it's a directory
        if os.path.isdir(label_path):
            # Count the number of videos in the folder
            videos = [f for f in os.listdir(label_path) if f.endswith(('.mp4', '.avi'))]
            label_counts[label_folder] = len(videos)

            # Print label and some example videos
            print(f"Label: {label_folder} | Videos: {len(videos)}")
            if videos:
                print(f"  Examples: {videos[:3]}")

    return label_counts

# Example usage
if __name__ == "__main__":
    base_path = "../../SPHAR-Dataset/videos" 
    label_data = explore_video_folder(base_path)

    print("\nSummary:")
    for label, count in label_data.items():
        print(f"  {label}: {count} videos")
