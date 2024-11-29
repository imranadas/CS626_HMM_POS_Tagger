import nltk
from nltk.corpus import brown
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from sklearn.model_selection import train_test_split
import json
import pickle
import streamlit as st
from datetime import timedelta
import re
import pandas as pd

class HMMPOSTagger:
    def __init__(self, smoothing=1.0):
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.tag_counts = defaultdict(int)
        self.word_counts = defaultdict(int)
        self.vocabulary = set()
        self.tags = set()
        self.smoothing = smoothing
        self.word_to_idx = {}
        self.tag_to_idx = {}
        self.idx_to_tag = {}
        self.transition_matrix = None
        self.emission_matrix = None
        
    def initialize_mappings(self, train_data):
        """Initialize word and tag mappings"""
        # Build vocabulary and tag sets
        for sent in train_data:
            for word, tag in sent:
                self.vocabulary.add(word.lower())
                self.tags.add(tag)
        
        # Create mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(self.tags)}
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}
        
    def compute_transition_matrix(self, train_data):
        """Compute transition probability matrix with Laplace smoothing"""
        num_tags = len(self.tags)
        transition_counts = np.zeros((num_tags, num_tags))
        tag_counts = np.zeros(num_tags)
        
        for sent in train_data:
            for i in range(1, len(sent)):
                prev_tag = self.tag_to_idx[sent[i - 1][1]]
                curr_tag = self.tag_to_idx[sent[i][1]]
                transition_counts[prev_tag, curr_tag] += 1
                tag_counts[prev_tag] += 1
        
        # Apply Laplace smoothing
        self.transition_matrix = np.log((transition_counts + self.smoothing) / 
                                      (tag_counts[:, None] + self.smoothing * num_tags))
    
    def compute_emission_matrix(self, train_data):
        """Compute emission probability matrix with Laplace smoothing"""
        num_tags = len(self.tags)
        num_words = len(self.vocabulary)
        emission_counts = np.zeros((num_tags, num_words))
        tag_counts = np.zeros(num_tags)
        
        for sent in train_data:
            for word, tag in sent:
                word_idx = self.word_to_idx[word.lower()]
                tag_idx = self.tag_to_idx[tag]
                emission_counts[tag_idx, word_idx] += 1
                tag_counts[tag_idx] += 1
        
        # Apply Laplace smoothing
        self.emission_matrix = np.log((emission_counts + self.smoothing) / 
                                    (tag_counts[:, None] + self.smoothing * num_words))
    
    def train(self, tagged_sentences, progress_bar=None, status_text=None):
        """Train the HMM POS tagger"""
        if not tagged_sentences:
            logging.warning("Training data is empty. Aborting training.")
            return
        
        total_sentences = len(tagged_sentences)
        start_time = time.time()
        
        if progress_bar is None:
            progress_bar = st.progress(0)
        if status_text is None:
            status_text = st.empty()
        
        # Phase 1: Initialize mappings
        status_text.text("Phase 1/3: Initializing mappings...")
        self.initialize_mappings(tagged_sentences)
        progress_bar.progress(0.33)
        
        # Phase 2: Compute transition probabilities
        status_text.text("Phase 2/3: Computing transition probabilities...")
        self.compute_transition_matrix(tagged_sentences)
        progress_bar.progress(0.66)
        
        # Phase 3: Compute emission probabilities
        status_text.text("Phase 3/3: Computing emission probabilities...")
        self.compute_emission_matrix(tagged_sentences)
        progress_bar.progress(1.0)
        
        total_time = time.time() - start_time
        status_text.text(f"Training completed in {timedelta(seconds=int(total_time))}!")
    
    def viterbi_improved(self, sentence):
        """Enhanced Viterbi algorithm implementation"""
        sentence = [word.lower() for word in sentence]
        num_tags = len(self.tags)
        len_sent = len(sentence)
        
        # Initialize matrices
        viterbi_matrix = np.full((num_tags, len_sent), -np.inf)
        backpointer = np.zeros((num_tags, len_sent), dtype=int)
        
        # Initialize first column
        word_idx = self.word_to_idx.get(sentence[0], -1)
        if word_idx != -1:
            for tag_idx in range(num_tags):
                viterbi_matrix[tag_idx, 0] = self.emission_matrix[tag_idx, word_idx]
        else:
            # Handle unknown word
            morph_tag = self.find_state_by_morphology(sentence[0])
            morph_tag_idx = self.tag_to_idx.get(morph_tag)
            if morph_tag_idx is not None:
                viterbi_matrix[morph_tag_idx, 0] = 0  # Log probability of 1
        
        # Recursion
        for t in range(1, len_sent):
            word_idx = self.word_to_idx.get(sentence[t], -1)
            for curr_tag in range(num_tags):
                if word_idx != -1:
                    emission_prob = self.emission_matrix[curr_tag, word_idx]
                else:
                    # Handle unknown word
                    morph_tag = self.find_state_by_morphology(sentence[t])
                    emission_prob = 0 if self.idx_to_tag[curr_tag] == morph_tag else -np.inf
                
                prob_transitions = viterbi_matrix[:, t - 1] + self.transition_matrix[:, curr_tag] + emission_prob
                viterbi_matrix[curr_tag, t] = np.max(prob_transitions)
                backpointer[curr_tag, t] = np.argmax(prob_transitions)
        
        # Termination
        best_path = []
        best_last_tag = np.argmax(viterbi_matrix[:, -1])
        best_path.append(best_last_tag)
        
        # Backtrace
        for t in range(len_sent - 1, 0, -1):
            best_last_tag = backpointer[best_last_tag, t]
            best_path.insert(0, best_last_tag)
        
        return [self.idx_to_tag[tag_idx] for tag_idx in best_path]
    
    def find_state_by_morphology(self, word):
        """Use morphological cues to predict POS tag for unknown words"""
        if re.search(r'.*(ing|ed|es|ould)$', word.lower()):
            return 'VERB'
        elif re.search(r'to$', str(word).lower()):
            return 'PRT'
        elif re.search(r'^-?[0-9]+(.[0-9]+)?\\.*$', str(word).lower()):
            return 'NUM'
        elif re.search(r'.*\'s$', word.lower()):
            return 'NOUN'
        elif re.search(r'.*ness$', word.lower()):
            return 'NOUN'
        elif re.search(r'(The|the|A|a|An|an)$', word):
            return 'DET'
        elif re.search(r'.*able$', word.lower()):
            return 'ADJ'
        elif re.search(r'.*ly$', word.lower()):
            return 'ADV'
        elif re.search(r'(He|he|She|she|It|it|I|me|Me|You|you|His|his|Her|her|Its|its|my|Your|your|Yours|yours)$', word):
            return 'PRON'
        elif re.search(r'(on|On|at|At|since|Since|For|for|Ago|ago|before|Before|till|Till|until|Until|by|By|Beside|beside|under|Under|below|Below|over|Over|above|Above|across|Across|Through|through|Into|into|towards|Towards|onto|Onto|from|From)$', word):
            return 'ADP'
        elif re.search(r'(\'|\"|\.|\(|\)|\?|\[|\]|\:|\;)+', word):
            return '.'
        else:
            return 'NOUN'

    def save_model(self, filename):
        """Save model to file"""
        model_dict = {
            'transition_matrix': self.transition_matrix,
            'emission_matrix': self.emission_matrix,
            'vocabulary': list(self.vocabulary),
            'tags': list(self.tags),
            'word_to_idx': self.word_to_idx,
            'tag_to_idx': self.tag_to_idx,
            'idx_to_tag': self.idx_to_tag,
            'smoothing': self.smoothing
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_dict, f)
        logging.info(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename):
        """Load model from file"""
        with open(filename, 'rb') as f:
            model_dict = pickle.load(f)
            tagger = HMMPOSTagger(smoothing=model_dict['smoothing'])
            tagger.transition_matrix = model_dict['transition_matrix']
            tagger.emission_matrix = model_dict['emission_matrix']
            tagger.vocabulary = set(model_dict['vocabulary'])
            tagger.tags = set(model_dict['tags'])
            tagger.word_to_idx = model_dict['word_to_idx']
            tagger.tag_to_idx = model_dict['tag_to_idx']
            tagger.idx_to_tag = model_dict['idx_to_tag']
            return tagger

def save_evaluation_results(y_true, y_pred, tags):
    """Save all evaluation results to files"""
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(tags))
    np.save('confusion_matrix.npy', cm)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(tags), 
                yticklabels=list(tags))
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate per-POS metrics
    per_pos_metrics = {}
    for i, tag in enumerate(tags):
        true_binary = [1 if t == tag else 0 for t in y_true]
        pred_binary = [1 if p == tag else 0 for p in y_pred]
        precision, recall, f1, _ = precision_recall_fscore_support(true_binary, pred_binary, average='binary')
        per_pos_metrics[tag] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    # Save per-POS metrics
    with open('per_pos_performance_metrics.json', 'w') as f:
        json.dump(per_pos_metrics, f, indent=4)
    
    # Calculate overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    overall_metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true))
    }
    
    # Save overall metrics
    with open('overall_performance_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    
    # Find most mismatched tags
    mismatches = []
    for i, tag1 in enumerate(tags):
        for j, tag2 in enumerate(tags):
            if i != j:
                mismatches.append([float(cm[i, j]), str(tag1), str(tag2)])
    
    mismatches.sort(reverse=True)
    top_mismatches = mismatches[:10]  # Save top 10 mismatches
    
    # Save mismatched tags
    with open('most_mismatched_tags.json', 'w') as f:
        json.dump(top_mismatches, f, indent=4)

def train_and_evaluate():
    """Main training and evaluation function"""
    st.title("HMM POS Tagger Training")
    
    # Download NLTK resources
    with st.spinner("Downloading NLTK resources..."):
        nltk.download('brown')
        nltk.download('universal_tagset')
        nltk.download('punkt')
    
    # Load and split Brown corpus
    tagged_sents = brown.tagged_sents(tagset='universal')
    train_set, test_set = train_test_split(tagged_sents, test_size=0.05, random_state=42)
    
    st.success(f"Loaded {len(tagged_sents)} sentences from Brown corpus")
    st.info(f"Split into {len(train_set)} training and {len(test_set)} test sentences")
    
    # Create progress containers
    st.subheader("Training Progress")
    train_progress = st.progress(0)
    train_status = st.empty()
    
    # Initialize and train model
    tagger = HMMPOSTagger()
    tagger.train(train_set, progress_bar=train_progress, status_text=train_status)
    
    # Evaluate on test set
    st.subheader("Evaluation Progress")
    eval_progress = st.progress(0)
    eval_status = st.empty()
    
    # Collect predictions
    eval_status.text("Evaluating model on test set...")
    y_true = []
    y_pred = []
    
    for i, sent in enumerate(test_set):
        words, true_tags = zip(*sent)
        predicted_tags = tagger.viterbi_improved(words)
        
        y_true.extend(true_tags)
        y_pred.extend(predicted_tags)
        
        eval_progress.progress((i + 1) / len(test_set))
    
    # Save evaluation results
    with st.spinner("Saving evaluation results..."):
        save_evaluation_results(y_true, y_pred, tagger.tags)
    
    # Save the trained model
    tagger.save_model('hmm_pos_tagger.pkl')
    
    st.success("""Training and evaluation completed! Files saved:
    - confusion_matrix.npy
    - confusion_matrix.png
    - most_mismatched_tags.json
    - overall_performance_metrics.json
    - per_pos_performance_metrics.json
    - hmm_pos_tagger.pkl""")
    
    # Display overall metrics
    with open('overall_performance_metrics.json', 'r') as f:
        metrics = json.load(f)
        
    st.subheader("Overall Model Performance")
    metrics_df = pd.DataFrame([
        {"Metric": k.capitalize(), "Value": f"{v*100:.2f}%"}
        for k, v in metrics.items()
    ])
    st.table(metrics_df)

if __name__ == "__main__":
    st.set_page_config(
        page_title="HMM POS Tagger Training",
        layout="wide"
    )
    
    st.markdown("""
    # HMM POS Tagger Training Dashboard
    This application trains an improved Hidden Markov Model for Part-of-Speech tagging using the Brown corpus.
    Features include:
    - Morphological analysis for unknown words
    - Enhanced emission probability calculation
    - Improved rare word handling
    - Feature-based tagging
    - Complete model evaluation and metrics
    """)
    
    if st.button("Start Training"):
        try:
            train_and_evaluate()
        except Exception as e:
            st.error(f"An error occurred during training: {str(e)}")
            logging.error(f"Training error: {str(e)}", exc_info=True)
    else:
        st.info("Click 'Start Training' to begin the training process.")