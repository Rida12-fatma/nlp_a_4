import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model from Sentence-Transformers
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to get sentence embeddings
def get_sentence_embedding(sentence):
    # Get the embedding of the sentence
    embedding = model.encode([sentence])[0]
    return embedding

# Function to compute similarity and predict NLI
def predict_nli(premise, hypothesis):
    premise_embedding = get_sentence_embedding(premise)
    hypothesis_embedding = get_sentence_embedding(hypothesis)
    
    # Cosine similarity between the sentence embeddings
    similarity = cosine_similarity([premise_embedding], [hypothesis_embedding])[0][0]
    
    # Predicted NLI labels based on cosine similarity
    if similarity > 0.7:
        return "Entailment"
    elif similarity > 0.3:
        return "Neutral"
    else:
        return "Contradiction"

# Streamlit UI for input and displaying results
st.set_page_config(page_title="NLI Prediction", page_icon="🔍")

st.markdown("<h1 style='text-align: center; color: blue;'>Text Similarity & NLI Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter two sentences below to predict their relationship (Entailment, Neutral, or Contradiction).</p>", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    premise = st.text_area("Premise Sentence", height=150)
with col2:
    hypothesis = st.text_area("Hypothesis Sentence", height=150)

st.markdown("---")

if st.button("Predict Relationship", use_container_width=True):
    if premise and hypothesis:
        result = predict_nli(premise, hypothesis)
        st.markdown(f"<h3 style='text-align: center; color: green;'>Prediction: {result}</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align: center; color: red;'>Please enter both sentences for prediction.</h3>", unsafe_allow_html=True)
