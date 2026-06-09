import pickle
import numpy as np
import tensorflow as tf
import cv2

from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

print("Loading embeddings...")

feature_list = np.array(
    pickle.load(open('embeddings.pkl', 'rb'))
)

filenames = pickle.load(
    open('filenames.pkl', 'rb')
)

print("Embeddings Shape:", feature_list.shape)
print("Number of Images:", len(filenames))

model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

print("Model Loaded Successfully!")

img = image.load_img(
    'sample/jersey.jpg',
    target_size=(224, 224)
)

img_array = image.img_to_array(img)

expanded_img_array = np.expand_dims(
    img_array,
    axis=0
)

preprocessed_img = preprocess_input(
    expanded_img_array
)

result = model.predict(
    preprocessed_img,
    verbose=0
).flatten()

normalized_result = result / norm(result)

print("Feature Vector Shape:")
print(normalized_result.shape)

print("Building KNN Model...")

neighbors = NearestNeighbors(
    n_neighbors=5,
    algorithm='brute',
    metric='euclidean'
)

neighbors.fit(feature_list)

print("KNN Ready!")

distances, indices = neighbors.kneighbors(
    [normalized_result]
)

print("\nRecommended Image Indexes:")
print(indices)

print("\nRecommended Images:\n")

for file in indices[0]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', temp_img)

cv2.waitKey(0)