# Utility script for preprocessing LFW dataset
# https://vis-www.cs.umass.edu/lfw/

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from PIL import Image
from ultralytics import YOLO

LFW_PATH = "datasets/lfw-deepfunneled"
LFW_YOLO_PATH = "datasets/lfw-yolo"

def crop_faces():
    """
    Function for cropping faces from the LFW dataset This is because the faces in
    the LFW dataset are detected using the Viola-Jones face detector. We are using
    YOLO for face detection so we want to make sure that the faces data are what YOLO detects.
    """
    output_path = LFW_YOLO_PATH
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load YOLO model on the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model="models/yolov8l-face.pt", task="detect").to(device)

    for root, _, files in tqdm(
        os.walk(LFW_PATH),
        total=len(os.listdir(LFW_PATH)),
        desc="Using YOLO detect for the faces: ",
    ):
        if root == LFW_PATH:
            continue

        for file in files:
            old_path = os.path.join(root, file)

            # Load the image and run through YOLO
            img = Image.open(old_path)
            results = model(img, verbose=False)
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                    new_img = img.crop((x1, y1, x2, y2))

                    new_dir = os.path.join(output_path, os.path.basename(root))
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    new_path = os.path.join(new_dir, file)
                    new_img.save(new_path)


def main():
    # Clear any existing output CSV files first
    for file in os.listdir(LFW_YOLO_PATH):
        if file.endswith(".csv"):
            os.remove(os.path.join(LFW_YOLO_PATH, file))

    # Initialize an empty list to store the image pairs
    face_pairs = []
    names = os.listdir(LFW_YOLO_PATH)

    # Count the similar and dissimilar pairs
    similar_pairs = 0
    dissimilar_pairs = 0

    # Iterate through the images in the LFW dataset
    for root, _, files in tqdm(
        os.walk(LFW_YOLO_PATH), total=len(names), desc="Getting image pairs: "
    ):
        # Skip the root directory
        if root == LFW_YOLO_PATH:
            continue

        # Gather similar images if there are any
        if len(files) > 1:
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    # Get the path to the images
                    img1_path = os.path.join(root, files[i])
                    img2_path = os.path.join(root, files[j])
                    # Append the image pair to the list
                    face_pairs.append((img1_path, img2_path, 1))

                    # Increment the count of similar pairs
                    similar_pairs += 1

        # Gather dissimilar images by sampling from other folders
        # Make sure to avoid sampling the same person
        names_copy = names.copy()
        names_copy.remove(os.path.basename(root))

        # Get images from 50 other different people
        for _ in range(50):
            # Sample the name of the person
            root2 = np.random.choice(names_copy, 1)[0]

            # Get the path to the images
            img1_path = os.path.join(root, files[0])
            img2_path = os.path.join(
                LFW_YOLO_PATH,
                root2,
                np.random.choice(os.listdir(os.path.join(LFW_YOLO_PATH, root2)), 1)[0],
            )
            # Append the image pair to the list
            face_pairs.append((img1_path, img2_path, 0))

            # Increment the count of dissimilar pairs
            dissimilar_pairs += 1

    print(f"Similar pairs: {similar_pairs}")
    print(f"Dissimilar pairs: {dissimilar_pairs}")

    # Create a pandas DataFrame from the list of image pairs
    df = pd.DataFrame(face_pairs, columns=["img1", "img2", "label"])

    # Save the DataFrame to a CSV file
    csv_output_path = os.path.join(LFW_YOLO_PATH, "image_pairs.csv")
    df.to_csv(csv_output_path, index=False)


if __name__ == "__main__":
    crop_faces()
    main()
    
