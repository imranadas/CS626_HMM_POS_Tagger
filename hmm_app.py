import streamlit as st
from hmm_training import HMMPOSTagger
import nltk
from nltk.tokenize import word_tokenize
from PIL import Image
import pandas as pd
import json

st.set_page_config(page_title="HMM POS App")

# Download necessary NLTK resources
nltk.download('punkt')

# Load the trained model
@st.cache_resource
def load_tagger(model_path='hmm_pos_tagger.pkl'):
    try:
        tagger = HMMPOSTagger.load_model(model_path)
        st.success("Model loaded successfully!")
        return tagger
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
tagger = load_tagger()

# Load JSON data from file
def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

@st.cache_data
def load_test_data(json_file):
    try:
        with open(json_file, 'r') as f:
            test_data = json.load(f)
        st.success("Test data loaded successfully!")
        return test_data
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return []
    
def tokenize_sentences(test_data):
    return [(word_tokenize(item['sentence']), item['tags']) for item in test_data]

# Load and display confusion matrix image
def display_confusion_matrix(image_path='confusion_matrix.png'):
    try:
        image = Image.open(image_path)
        st.image(image, caption="Confusion Matrix", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")

# Convert JSON data to DataFrame for display
def display_json_as_table(json_data, title):
    if json_data:
        st.subheader(title)
        df = pd.DataFrame.from_dict(json_data, orient='index')
        st.table(df)

# Display most mismatched tags in descending order
def display_most_mismatched_tags(data):
    if data:
        st.subheader("Most Mismatched Tags")
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=["Count", "True Tag", "Predicted Tag"])
        # Sort DataFrame by count in descending order
        df_sorted = df.sort_values(by="Count", ascending=False)
        # Display sorted DataFrame
        st.table(df_sorted)  # Use st.table to display the sorted mismatches
        
def evaluate_accuracy(tagger, tokenized_data):
    total_tokens = 0
    correct_tokens = 0

    for sentence, true_tags in tokenized_data:
        predicted_tags = tagger.viterbi(sentence)  # Predict using loaded model

        # Compare predicted and true tags
        for predicted_tag, true_tag in zip(predicted_tags, true_tags):
            if predicted_tag == true_tag:
                correct_tokens += 1
            total_tokens += 1

    # Calculate accuracy
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return accuracy



st.title("HMM POS Tagger")

if tagger:
    st.subheader("Enter a sentence to tag:")
    user_input = st.text_area("Input Sentence", value="This is a test sentence.", height=100)

    if st.button("Tag Sentence"):
        if user_input.strip():
            # Tokenize input sentence
            tokens = word_tokenize(user_input.lower())
            # Tag the sentence using the loaded model
            tagged_sentence = tagger.viterbi(tokens)
            
            # Display the tagged sentence in tabular format
            st.write("Tagged Sentence:")
            tagged_df = pd.DataFrame(list(zip(tokens, tagged_sentence)), columns=['Word', 'Predicted Tag'])
            st.table(tagged_df)  # Use st.table to display the tagged sentence as a table
        else:
            st.warning("Please enter a valid sentence.")

st.write("---")
st.subheader("Performance Metrics")

# Display Confusion Matrix Image
display_confusion_matrix()

# Display Overall Performance Metrics
overall_metrics = load_json('overall_performance_metrics.json')
display_json_as_table(overall_metrics, "Overall Performance Metrics")

# Display Per POS Performance Metrics
per_pos_metrics = load_json('per_pos_performance_metrics.json')
display_json_as_table(per_pos_metrics, "Per POS Performance Metrics")

# Display Most Mismatched Tags
most_mismatched_tags = load_json('most_mismatched_tags.json')
display_most_mismatched_tags(most_mismatched_tags)

if st.button("Analyze Test Set"):
    if tagger:
        # Load the test data
        test_data = load_test_data('test_data.json')
        
        if test_data:
            # Tokenize the test sentences
            tokenized_test_data = tokenize_sentences(test_data)

            # Evaluate the accuracy
            accuracy = evaluate_accuracy(tagger, tokenized_test_data)
            st.write(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
        else:
            st.error("Test data could not be loaded.")

st.write("---")
st.info("Developed using HMM POS Tagger Model")