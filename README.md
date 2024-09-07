# HMM POS Tagger

This project implements a Hidden Markov Model (HMM) for Part-of-Speech (POS) tagging using Python. It includes training the HMM model, evaluating it, and creating a Streamlit web application to interact with the model.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup](#setup)
3. [Running the Code](#running-the-code)
4. [Using the Streamlit App](#using-the-streamlit-app)
5. [File Descriptions](#file-descriptions)

## Prerequisites

Ensure you have the following software installed:

- Python 3.7 or later
- pip (Python package installer)

You will also need to install the following Python libraries:

- `nltk`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`
- `streamlit`
- `altair`
- `Pillow` (for image handling)

You can install these libraries using pip:

```bash
pip install nltk numpy scikit-learn matplotlib seaborn pandas streamlit altair Pillow
```

## Setup

1. **Clone the Repository**

   Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/hmm-pos-tagger.git
   cd hmm-pos-tagger
   ```

2. **Download NLTK Resources**

   The code requires certain NLTK resources. They are downloaded automatically when running the code, but you can manually download them if needed:

   ```python
   import nltk
   nltk.download('brown')
   nltk.download('universal_tagset')
   nltk.download('punkt')
   ```

## Running the Code

1. **Train the Model**

   Run the `hmm_training.py` script to train the HMM POS tagger model and generate performance metrics:

   ```bash
   python hmm_training.py
   ```

   This will create the following files:

   - `confusion_matrix.npy`
   - `confusion_matrix.png`
   - `most_mismatched_tags.json`
   - `overall_performance_metrics.json`
   - `per_pos_performance_metrics.json`
   - `hmm_pos_tagger.pkl` (the trained model)

2. **Run the Streamlit App**

   Start the Streamlit application to interact with the trained model:

   ```bash
   streamlit run app.py
   ```

   This will open a web browser with the Streamlit app where you can input sentences and view the tagged sentences, performance metrics, and visualizations.

## Using the Streamlit App

- **Tagging Sentences**:
  Enter a sentence in the provided text area and click the "Tag Sentence" button to see the sentence tagged with POS tags.

- **Viewing Performance Metrics**:
  The app displays various performance metrics and visualizations:
  - **Confusion Matrix**: An image showing the confusion matrix of the model.
  - **Overall Performance Metrics**: JSON data showing average precision, recall, F1 score, etc.
  - **Per POS Performance Metrics**: JSON data showing performance metrics for each POS tag.
  - **Most Mismatched Tags**: A table showing the most frequently confused POS tag pairs, sorted by mismatch count.

## File Descriptions

- **`hmm_training.py`**: Script for training the HMM POS tagger, evaluating its performance, and saving the model and metrics.
- **`app.py`**: Streamlit application for interacting with the trained model and displaying results.
- **`confusion_matrix.npy`**: NumPy file containing the confusion matrix data.
- **`confusion_matrix.png`**: Image file of the confusion matrix.
- **`most_mismatched_tags.json`**: JSON file with the most mismatched POS tag pairs.
- **`overall_performance_metrics.json`**: JSON file with overall performance metrics.
- **`per_pos_performance_metrics.json`**: JSON file with performance metrics for each POS tag.
- **`hmm_pos_tagger.pkl`**: Pickle file containing the trained HMM POS tagger model.

## Troubleshooting

- **Model Loading Issues**: Ensure the `hmm_pos_tagger.pkl` file is correctly created and not corrupted.
- **Library Errors**: Verify all required libraries are installed and up-to-date.