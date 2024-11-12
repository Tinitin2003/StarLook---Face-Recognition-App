import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import cv2
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from PIL import Image
IMAGE_SIZE = [224, 224]
from keras.applications import VGG16
# Load the VGG16 model with pre-trained weights and exclude the top layers
from keras_facenet import FaceNet
embedder = FaceNet()


feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

detector = MTCNN()
# load img -> face detection
sample_img = cv2.imread('sample/nitin.jpg')
results = detector.detect_faces(sample_img)

if len(results) > 0:
    x, y, width, height = results[0]['box']
    face = sample_img[y:y+height,x:x+width]
else:
    print("No faces detected")
    face=sample_img

cv2.imshow('input',face)
#  extract its features
image = Image.fromarray(face)
image = image.resize((224,224))

face_array = np.asarray(image)

face_array = face_array.astype('float32')

expanded_img = np.expand_dims(face_array,axis=0)
result = embedder.embeddings(expanded_img)
#print(result)
#print(result.shape)
# find the cosine distance of current image with all the 8655 features
similarity = []
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])

index_pos = sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]
print(filenames[index_pos])
temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('output',temp_img)
cv2.waitKey(0)
