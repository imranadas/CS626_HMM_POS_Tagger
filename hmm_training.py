import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
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
from datetime import datetime, timedelta
import re
from pathlib import Path
import pandas as pd

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hmm_training.log')
    ]
)

class HMMPOSTagger:
    def __init__(self, smoothing=1e-5):
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.tag_counts = defaultdict(int)
        self.word_counts = defaultdict(int)
        self.vocabulary = set()
        self.tags = set()
        self.smoothing = smoothing
        self.unknown_word_token = '<UNK>'
        self.rare_word_threshold = 5
        self.suffix_length = 3
        self.prefix_length = 2
        self.suffix_dict = defaultdict(lambda: defaultdict(int))
        self.prefix_dict = defaultdict(lambda: defaultdict(int))
        
    def get_word_features(self, word):
        """Extract features from a word to help with unknown word handling"""
        features = []
        word = word.lower()
        
        if any(c.isdigit() for c in word):
            features.append('HAS_NUMBER')
        if word[0].isupper():
            features.append('CAPITALIZED')
        if word.isupper():
            features.append('ALL_CAPS')
        if '-' in word:
            features.append('HAS_HYPHEN')
            
        suffix = word[-self.suffix_length:] if len(word) > self.suffix_length else word
        features.append(f'SUFFIX_{suffix}')
        
        prefix = word[:self.prefix_length] if len(word) > self.prefix_length else word
        features.append(f'PREFIX_{prefix}')
        
        return features

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
        elif re.search(r'(\'|\"|\.|\(|\)|\?|\[|\]|\:|\;)+',word):
            return '.'
        else:
            return 'NOUN'
    
    def train(self, tagged_sentences, progress_bar=None, status_text=None):
        """Train the HMM POS tagger with progress visualization"""
        if not tagged_sentences:
            logging.warning("Training data is empty. Aborting training.")
            return
        
        total_sentences = len(tagged_sentences)
        start_time = time.time()
        
        if progress_bar is None:
            progress_bar = st.progress(0)
        if status_text is None:
            status_text = st.empty()
        
        # Phase 1: Count word frequencies
        status_text.text("Phase 1/3: Counting word frequencies...")
        word_freq = defaultdict(int)
        for i, sent in enumerate(tagged_sentences):
            for word, _ in sent:
                word = word.lower()
                word_freq[word] += 1
            
            if i % 100 == 0:
                progress = (i + 1) / total_sentences
                progress_bar.progress(progress / 3)
                
        # Create rare word vocabulary
        rare_words = {word for word, freq in word_freq.items() if freq < self.rare_word_threshold}
        
        # Phase 2: Collect transition and emission counts
        status_text.text("Phase 2/3: Collecting transition and emission counts...")
        for i, sent in enumerate(tagged_sentences):
            prev_tag = '<START>'
            for word, tag in sent:
                word = word.lower()
                
                # Handle transition probabilities
                self.transition_probs[prev_tag][tag] += 1
                
                # Handle emission probabilities
                if word in rare_words:
                    features = self.get_word_features(word)
                    for feature in features:
                        self.emission_probs[tag][feature] += 1
                else:
                    self.emission_probs[tag][word] += 1
                
                # Update counts and sets
                self.tag_counts[tag] += 1
                self.word_counts[word] += 1
                self.vocabulary.add(word)
                self.tags.add(tag)
                
                # Update suffix and prefix dictionaries
                suffix = word[-self.suffix_length:] if len(word) > self.suffix_length else word
                prefix = word[:self.prefix_length] if len(word) > self.prefix_length else word
                self.suffix_dict[suffix][tag] += 1
                self.prefix_dict[prefix][tag] += 1
                
                prev_tag = tag
            
            # Handle end of sentence
            self.transition_probs[prev_tag]['<END>'] += 1
            
            if i % 100 == 0:
                progress = (i + 1) / total_sentences
                progress_bar.progress((1 + progress) / 3)
        
        # Phase 3: Compute probabilities
        status_text.text("Phase 3/3: Computing probabilities...")
        self.compute_probabilities()
        progress_bar.progress(1.0)
        
        total_time = time.time() - start_time
        status_text.text(f"Training completed in {timedelta(seconds=int(total_time))}!")
    
    def compute_probabilities(self):
        """Compute transition and emission probabilities with smoothing"""
        # Compute transition probabilities
        for prev_tag in self.transition_probs:
            total = sum(self.transition_probs[prev_tag].values()) + self.smoothing * (len(self.tags) + 1)
            for tag in self.tags | {'<END>'}:
                count = self.transition_probs[prev_tag][tag]
                self.transition_probs[prev_tag][tag] = (count + self.smoothing) / total
        
        # Compute emission probabilities
        for tag in self.emission_probs:
            total = sum(self.emission_probs[tag].values()) + self.smoothing * len(self.vocabulary)
            for word in self.vocabulary:
                count = self.emission_probs[tag][word]
                suffix = word[-self.suffix_length:] if len(word) > self.suffix_length else word
                prefix = word[:self.prefix_length] if len(word) > self.prefix_length else word
                
                suffix_prob = (self.suffix_dict[suffix][tag] + self.smoothing) / (self.tag_counts[tag] + self.smoothing * len(self.suffix_dict))
                prefix_prob = (self.prefix_dict[prefix][tag] + self.smoothing) / (self.tag_counts[tag] + self.smoothing * len(self.prefix_dict))
                
                word_prob = (count + self.smoothing) / total
                self.emission_probs[tag][word] = 0.7 * word_prob + 0.15 * suffix_prob + 0.15 * prefix_prob
    
    def viterbi_improved(self, sentence):
        """Enhanced Viterbi algorithm with unknown word handling"""
        V = [{}]
        path = {}
        tags = list(self.tags)
        
        # Process first word
        word = sentence[0].lower()
        features = self.get_word_features(word) if word not in self.vocabulary else []
        
        for tag in tags:
            # Enhanced emission probability calculation
            if word in self.vocabulary:
                emission_prob = self.emission_probs[tag].get(word, self.smoothing)
            else:
                # For unknown words, combine morphological and feature-based probabilities
                morph_tag = self.find_state_by_morphology(word)
                if tag == morph_tag:
                    emission_prob = 0.8
                else:
                    feature_probs = [self.emission_probs[tag].get(feature, 0) for feature in features]
                    emission_prob = sum(feature_probs) / len(features) if features else self.smoothing
            
            V[0][tag] = self.transition_probs['<START>'][tag] * emission_prob
            path[tag] = [tag]
        
        # Run Viterbi for remaining words
        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}
            word = sentence[t].lower()
            features = self.get_word_features(word) if word not in self.vocabulary else []
            
            for tag in tags:
                if word in self.vocabulary:
                    emission_prob = self.emission_probs[tag].get(word, self.smoothing)
                else:
                    morph_tag = self.find_state_by_morphology(word)
                    if tag == morph_tag:
                        emission_prob = 0.8
                    else:
                        feature_probs = [self.emission_probs[tag].get(feature, 0) for feature in features]
                        emission_prob = sum(feature_probs) / len(features) if features else self.smoothing
                
                (prob, state) = max(
                    (V[t-1][prev_tag] * self.transition_probs[prev_tag][tag] * emission_prob, prev_tag)
                    for prev_tag in tags
                )
                V[t][tag] = prob
                new_path[tag] = path[state] + [tag]
            path = new_path
        
        # End case
        n = len(sentence) - 1
        (prob, state) = max((V[n][tag] * self.transition_probs[tag]['<END>'], tag) for tag in tags)
        
        return path[state]

    def save_model(self, filename):
        """Save model to file"""
        model_dict = {
            'transition_probs': dict(self.transition_probs),
            'emission_probs': dict(self.emission_probs),
            'tag_counts': dict(self.tag_counts),
            'word_counts': dict(self.word_counts),
            'vocabulary': list(self.vocabulary),
            'tags': list(self.tags),
            'smoothing': self.smoothing,
            'unknown_word_token': self.unknown_word_token,
            'suffix_dict': dict(self.suffix_dict),
            'prefix_dict': dict(self.prefix_dict)
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
            tagger.transition_probs = defaultdict(lambda: defaultdict(float), model_dict['transition_probs'])
            tagger.emission_probs = defaultdict(lambda: defaultdict(float), model_dict['emission_probs'])
            tagger.tag_counts = defaultdict(int, model_dict['tag_counts'])
            tagger.word_counts = defaultdict(int, model_dict['word_counts'])
            tagger.vocabulary = set(model_dict['vocabulary'])
            tagger.tags = set(model_dict['tags'])
            tagger.unknown_word_token = model_dict['unknown_word_token']
            tagger.suffix_dict = defaultdict(lambda: defaultdict(int), model_dict['suffix_dict'])
            tagger.prefix_dict = defaultdict(lambda: defaultdict(int), model_dict['prefix_dict'])
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