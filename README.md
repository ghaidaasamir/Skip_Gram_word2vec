# Word2Vec Skip-Gram Model

Implementation of a Word2Vec model using the skip-gram in TensorFlow.

## Approach

  - **Data Download & Preprocessing:**  
    Download the Text8 dataset, extract and preprocess the text.

  - **Subsampling:**  
    Remove high-frequency words that add noise.

  - **Batch Generation:**  
    Create batches for training where for each center word, target context words are extracted from a variable window size.

  - **Model Building:**  
    Embedding lookup, and negative sampling to approximate the softmax over a large vocabulary.

  - **Training & Validation:**  
    Train the network.

  - **Visualization:**  
    Use T-SNE to project the high-dimensional word embeddings into two dimensions to visualize clusters of similar words.

## Requirements

- **Python 3.6+**
- **TensorFlow 1.x**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **scikit-learn**
