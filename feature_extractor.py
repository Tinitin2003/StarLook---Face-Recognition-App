import numpy as np
import pickle
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input  # Preprocessing for VGG-like models
from keras_facenet import FaceNet
model = FaceNet()

# Load the list of filenames
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the pre-trained VGGFace model from the .h5 file
# model = load_model('vgg_face_model.h5')
# model.load_weights('vgg_face_weights.h5')

def feature_extractor(img_path, model):
    # Load image and preprocess it for VGG-like models
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)  # Use preprocess_input for VGG-based models

    # Extract features using the model
    yhat= model.embeddings(preprocessed_img)
    print(yhat[0])
    return yhat[0] # 512D image (1x1x512)


# Extract features for each image
features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file,model))

# Save the features to a pickle file
pickle.dump(features, open('embedding.pkl', 'wb'))
