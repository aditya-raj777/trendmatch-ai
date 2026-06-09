# TrendMatch AI 👕✨

A fashion recommendation system that finds visually similar clothing items using Deep Learning and Computer Vision.

## About the Project

TrendMatch AI allows users to upload a fashion image and receive recommendations for visually similar products from a catalogue of over 44,000 fashion items.

The system uses a pre-trained ResNet50 model to extract image features and K-Nearest Neighbors (KNN) for similarity search. A Streamlit web application provides an easy-to-use interface for interacting with the recommendation engine.

## Features

* Upload a fashion image
* Find visually similar products
* Deep learning based feature extraction using ResNet50
* Fast similarity search using KNN
* Interactive Streamlit web interface
* Supports a catalogue of 44,441 fashion products

## Tech Stack

* Python
* TensorFlow / Keras
* ResNet50
* Scikit-Learn
* NumPy
* OpenCV
* Streamlit

## Dataset Information

* Total Images: 44,441
* Feature Dimensions: 2048
* Feature Extraction Model: ResNet50

## Project Workflow

1. Upload an image
2. Extract image features using ResNet50
3. Generate a 2048-dimensional embedding
4. Compare with stored embeddings
5. Retrieve the most similar fashion items using KNN
6. Display recommendations

## Screenshots

### Home Page

<img width="661" height="963" alt="Front Page" src="https://github.com/user-attachments/assets/adf8fb63-cc21-4ea3-8cfe-d740fe8363c0" />


### Upload Image

<img width="347" height="722" alt="Recomendation  full" src="https://github.com/user-attachments/assets/6eee65bf-b16f-4ff0-9cff-870c199f49a9" />

<img width="667" height="970" alt="visual search" src="https://github.com/user-attachments/assets/8f7503c2-390e-442e-9089-fd5a489cd1fb" />

### Recommendations

<img width="1038" height="682" alt="recomendation" src="https://github.com/user-attachments/assets/09acb11e-5b90-4cbf-ab7c-b4a9d30bac03" />
<img width="771" height="1020" alt="recomendation by URL" src="https://github.com/user-attachments/assets/d59f1201-03d0-4138-ae40-91881692855b" />

### History Page 
<img width="661" height="967" alt="history page " src="https://github.com/user-attachments/assets/c31ab823-5aa5-4c85-ac0e-4e52ad3451bf" />

### Documentation Page 
<img width="1912" height="1021" alt="documentation page " src="https://github.com/user-attachments/assets/8612edb3-0f2c-45a7-9c11-b587c3a7f897" />


## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/trendmatch-ai.git
cd trendmatch-ai
```

Create a virtual environment:

```bash
python -m venv tfenv
```

Activate the environment:

```bash
tfenv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run main.py
```

## Note

The original dataset and generated embedding files are not included in this repository because of their large size. They can be generated locally using the feature extraction script.

Future Improvements

• Category-aware recommendations
• EfficientNetB0/EfficientNetB3 based feature extraction
• FAISS-based similarity retrieval for large-scale datasets
• Cloud deployment for public access
• User preference and personalization support

## Author

Aditya Raj

B.Tech Student | Machine Learning & Software Development Enthusiast
