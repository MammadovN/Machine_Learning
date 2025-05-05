# TensorFlow MovieLens Recommender

A simple movie recommendation system built with TensorFlow Recommenders (TFRS) on the MovieLens “ml-latest-small” dataset. Learns user–movie interactions to predict ratings and suggest top‐K movies per user.

---

## Features

- Download and preprocess MovieLens `ml-latest-small` dataset  
- Build user and movie embeddings with TensorFlow  
- Train a ranking (rating-prediction) model  
- Generate top-K movie recommendations for any user ID  

---

## Requirements

- Python 3.7+  
- TensorFlow 2.x  
- TensorFlow Recommenders  
- NumPy  
- Pandas  

---

## Installation
    ```bash
# (Optional) create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate      # Windows

# Install dependencies
    pip install tensorflow tensorflow-recommenders pandas numpy

---

## Dataset
# Download and unzip the MovieLens dataset:
    wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
    unzip ml-latest-small.zip
    Ensure `ratings.csv` and `movies.csv` are in your project root.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

