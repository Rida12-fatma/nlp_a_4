# nlp_a_4

# Text Similarity & NLI Prediction with Streamlit

This repository contains a Streamlit application for predicting the relationship between two sentences (Entailment, Neutral, or Contradiction) using Sentence Transformers. Additionally, this repository also includes code for training a BERT model for masked language modeling (MLM) on the BookCorpus dataset.

## Setup

### Requirements

Ensure you have the following dependencies installed:

- `streamlit>=1.10.0`
- `sentence-transformers>=2.2.0`
- `scikit-learn>=0.24.2`
- `torch>=1.10.0`

You can install these dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Running the Streamlit Application

To run the Streamlit application, execute the following command:

```bash
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser.

### BERT Model Training

The code for training a BERT model on the BookCorpus dataset is included in the `bert_training.py` script. The script trains a BERT model for masked language modeling (MLM) and saves the trained model weights to a file.

**Note:** The BERT model and its training script are too large to upload to GitHub, so the model weights file (`bert_mlm_weights.pth`) is not included in this repository. You can run the training script on your local machine to generate the model weights.

## Usage

### Streamlit Application

The Streamlit application allows you to enter two sentences and predict their relationship (Entailment, Neutral, or Contradiction) based on cosine similarity of their embeddings.

You can access the deployed Streamlit application at the following link: [Text Similarity & NLI Prediction App](https://a4-nlp-hwshcbbswict8fwghqdm9k.streamlit.app/)

### BERT Model Training

The `bert_training.py` script includes the following steps:
1. Loading the BookCorpus dataset.
2. Tokenizing and preprocessing the text data.
3. Training a BERT model for masked language modeling.
4. Saving the trained model weights.

To run the training script, execute:

```bash
python bert_training.py
```

## Files

- `app.py`: Streamlit application code.
- `bert_training.py`: BERT model training script.
- `requirements.txt`: List of required Python packages.
- `README.md`: This README file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
