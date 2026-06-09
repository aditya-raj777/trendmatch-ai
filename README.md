# TrendMatch AI

Deep Learning Based Fashion Recommendation System

## Overview

TrendMatch AI is a fashion recommendation platform that uses computer vision and deep learning to find visually similar fashion products.

The system extracts image embeddings using ResNet50 and performs similarity search using K-Nearest Neighbors (KNN).

## Features

* Visual similarity search
* ResNet50 feature extraction
* KNN recommendation engine
* Streamlit web application
* Dark luxury UI
* Real-time recommendations

## Tech Stack

* Python
* TensorFlow
* ResNet50
* Scikit-Learn
* Streamlit
* NumPy
* OpenCV

## Dataset

* 44,441 fashion product images
* 2048-dimensional image embeddings

## Project Workflow

Upload Image
→ Feature Extraction (ResNet50)
→ Embedding Generation
→ KNN Similarity Search
→ Recommended Fashion Products

## Installation

```bash
git clone <repository-url>
cd TrendMatch-AI

python -m venv tfenv
tfenv\Scripts\activate

pip install -r requirements.txt
streamlit run main.py
```

## Note

The dataset and generated embeddings are not included in this repository due to their large size.

Generate embeddings locally before running the application.

## Future Improvements

* EfficientNet-based embeddings
* Category-aware recommendations
* Similarity score visualization
* Cloud deployment

## Author

Aditya Raj
