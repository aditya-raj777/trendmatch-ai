import tensorflow as tf
import numpy as np
import os
import pickle

import pickle
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tqdm import tqdm

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load ResNet50 Model
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

print("✅ Model loaded successfully!")

# Feature Extraction Function
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    img_array = image.img_to_array(img)

    expanded_img_array = np.expand_dims(img_array, axis=0)

    preprocessed_img = preprocess_input(expanded_img_array)

    result = model.predict(
        preprocessed_img,
        verbose=0
    ).flatten()

    normalized_result = result / norm(result)

    return normalized_result


# Get All Image Paths
filenames = []

for file in os.listdir('images'):
    filenames.append(
        os.path.join('images', file)
    )

print(f"📁 Total Images Found: {len(filenames)}")

# Extract Features From ALL Images
feature_list = []

print("🚀 Starting feature extraction...")

for file in tqdm(filenames):
    try:
        feature_list.append(
            extract_features(file, model)
        )
    except Exception as e:
        print(f"\n❌ Error processing {file}")
        print(e)

print("\n✅ Feature extraction completed!")

feature_array = np.array(feature_list)

print("Feature Array Shape:")
print(feature_array.shape)

# Save Features
pickle.dump(
    feature_list,
    open('embeddings.pkl', 'wb')
)

pickle.dump(
    filenames,
    open('filenames.pkl', 'wb')
)

print("✅ embeddings.pkl saved")
print("✅ filenames.pkl saved")

feature_list = np.array(
    pickle.load(open('embeddings.pkl','rb'))
)

filenames = pickle.load(
    open('filenames.pkl','rb')
)

print(feature_list.shape)