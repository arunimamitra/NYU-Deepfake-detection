# -*- coding: utf-8 -*-
"""mtcnn_data_creation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14uFXcvSpUWnTmwdGHr2aWeO5vLp1urCK
"""

from facenet_pytorch import MTCNN
import os
from PIL import Image

# Function to get the face bounding box using MTCNN
def get_face_box(image_path, mtcnn):
    image = Image.open(image_path).convert('RGB')
    boxes, _ = mtcnn.detect(image)
    return boxes[0] if boxes is not None else None

# Initialize MTCNN
mtcnn = MTCNN(keep_all=True)

# Directory paths
train_deepfake_dir = "/scratch/dp3635/real_vs_fake/real-vs-fake/train/fake/"
output_dir = "/scratch/dp3635/cropped_real_vs_fake/real-vs-fake/train/fake/"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image in the directory
for filename in os.listdir(train_deepfake_dir):
    if filename.endswith(".jpg"):
        img_path = os.path.join(train_deepfake_dir, filename)

        # Get the bounding box of the face
        face_box = get_face_box(img_path, mtcnn)

        if face_box is not None:
            # Crop and save the face image
            image = Image.open(img_path).convert('RGB')
            cropped_face = image.crop(face_box)
            cropped_face.save(os.path.join(output_dir, filename))

print("Cropping completed. Cropped images are saved in:", output_dir)