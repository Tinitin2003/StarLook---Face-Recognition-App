import cv2
import pickle
from PIL import Image
import os
from tqdm import tqdm
from mtcnn import MTCNN

# Assuming you have a folder named 'images' containing the images
detector = MTCNN()
filenames = pickle.load(open('unclear_face.pkl', 'rb'))

for image_path in tqdm(filenames):

    try:
        # Read the image with error handling
        sample_img = cv2.imread(image_path)
        if sample_img is None:
            print(f"Error: Could not read image '{image_path}'. Skipping...")
            continue

        # Detect faces
        results = detector.detect_faces(sample_img)

        # Assuming only one face is detected per image
        if len(results) > 0:
            x, y, width, height = results[0]['box']

            # Extract the face
            face = sample_img[y:y + height, x:x + width]

            # Convert to PIL Image and resize
            image = Image.fromarray(face)

            # Save the extracted face with the same name in the same folder
            image.save(image_path)

    except Exception as e:
        print(f"Error processing image '{image_path}': {e}")