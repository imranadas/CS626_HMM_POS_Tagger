import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import KFold
import json
import pickle

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.StreamHandler(),  # Output to console
    logging.FileHandler('hmm_model_training.log')  # Save to file
])

# Download necessary NLTK resources
nltk.download('brown')
nltk.download('universal_tagset')
nltk.download('punkt')
nltk.download('punkt_tab')

class HMMPOSTagger:
    def __init__(self, smoothing=1e-3):
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.tag_counts = defaultdict(int)
        self.word_counts = defaultdict(int)
        self.vocabulary = set()
        self.tags = set()
        self.smoothing = smoothing  # Smoothing factor
        self.unknown_word_token = '<UNK>'
    
    def train(self, tagged_sentences):
        logging.info("Training HMM POS tagger...")
        if not tagged_sentences:
            logging.warning("Training data is empty. Aborting training.")
            return
        
        for idx, sent in enumerate(tagged_sentences):
            words, tags = zip(*sent)
            if idx % 1000 == 0:
                logging.info(f"Training on sentence {idx + 1}: {list(zip(words, tags))}")
            
            prev_tag = '<START>'
            for word, tag in sent:
                self.transition_probs[prev_tag][tag] += 1
                self.emission_probs[tag][word] += 1
                self.tag_counts[tag] += 1
                self.word_counts[word] += 1
                self.vocabulary.add(word)
                self.tags.add(tag)
                prev_tag = tag
            # Handle end of sentence
            self.transition_probs[prev_tag]['<END>'] += 1
        
        # Include '<UNK>' in vocabulary
        self.vocabulary.add(self.unknown_word_token)
        for tag in self.tags:
            self.emission_probs[tag][self.unknown_word_token] = self.smoothing

        # Convert counts to probabilities with smoothing
        for prev_tag in self.transition_probs:
            total = sum(self.transition_probs[prev_tag].values()) + self.smoothing * (len(self.tags) + 1)  # +1 for '<END>'
            for tag in self.transition_probs[prev_tag]:
                self.transition_probs[prev_tag][tag] = (self.transition_probs[prev_tag][tag] + self.smoothing) / total

        for tag in self.emission_probs:
            total = sum(self.emission_probs[tag].values()) + self.smoothing * len(self.vocabulary)
            for word in self.emission_probs[tag]:
                self.emission_probs[tag][word] = (self.emission_probs[tag][word] + self.smoothing) / total

        logging.info("Training completed.")
    
    def viterbi(self, sentence):
        logging.info(f"Tagging sentence: {sentence}")
        V = [{}]
        path = {}
        tags = list(self.tags)
        
        # Initialize base case (t == 0)
        for tag in tags:
            V[0][tag] = self.transition_probs['<START>'][tag] * self.emission_probs[tag].get(sentence[0], self.emission_probs[tag][self.unknown_word_token])
            path[tag] = [tag]

        # Run Viterbi for t > 0
        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}
            for tag in tags:
                (prob, state) = max(
                    (V[t - 1][prev_tag] * self.transition_probs[prev_tag][tag] * self.emission_probs[tag].get(sentence[t], self.emission_probs[tag][self.unknown_word_token]), prev_tag)
                    for prev_tag in tags
                )
                V[t][tag] = prob
                new_path[tag] = path[state] + [tag]
            path = new_path
        
        # End case
        n = len(sentence) - 1
        (prob, state) = max((V[n][tag] * self.transition_probs[tag]['<END>'], tag) for tag in tags)
        predicted_tags = path[state]
        logging.info(f"Predicted tags: {predicted_tags}")
        return predicted_tags

    def save_model(self, filename):
        # Convert defaultdicts to regular dicts for pickling
        model_dict = {
            'transition_probs': dict(self.transition_probs),
            'emission_probs': dict(self.emission_probs),
            'tag_counts': dict(self.tag_counts),
            'word_counts': dict(self.word_counts),
            'vocabulary': list(self.vocabulary),
            'tags': list(self.tags),
            'smoothing': self.smoothing,
            'unknown_word_token': self.unknown_word_token
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_dict, f)
        logging.info(f"Model saved to {filename}.")

    @staticmethod
    def load_model(filename):
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
            return tagger

def preprocess_data():
    logging.info("Preprocessing data...")
    brown_sents = brown.tagged_sents(tagset='universal')
    data = []
    for idx, sent in enumerate(brown_sents):
        words, tags = zip(*sent)
        # Tokenize words using word_tokenize
        tokens = [word_tokenize(word.lower()) for word in words]
        # Flatten list of tokenized words
        processed_sent = list(zip([item for sublist in tokens for item in sublist], tags))
        data.append(processed_sent)
        if idx % 1000 == 0:
            logging.info(f"Preprocessed {idx + 1} sentences.")
    logging.info(f"Preprocessed {len(data)} sentences.")
    return data

def evaluate_model(tagged_sentences, tagger, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_tags = list(set(tag for _, tag in sum(tagged_sentences, [])))

    overall_precision = []
    overall_recall = []
    overall_f1 = []
    overall_f05 = []
    overall_f2 = []

    per_tag_results = defaultdict(lambda: {'precision': [], 'recall': [], 'f1': []})
    conf_matrix = np.zeros((len(all_tags), len(all_tags)))

    fold_number = 1
    for train_index, test_index in kf.split(tagged_sentences):
        logging.info(f"Starting fold {fold_number}...")
        train_data = [tagged_sentences[i] for i in train_index]
        test_data = [tagged_sentences[i] for i in test_index]
        
        tagger.train(train_data)

        y_true = []
        y_pred = []

        for idx, sent in enumerate(test_data):
            words, true_tags = zip(*sent)
            predicted_tags = tagger.viterbi(words)
            if len(true_tags) != len(predicted_tags):
                logging.warning(f"Length mismatch: True tags ({len(true_tags)}) vs Predicted tags ({len(predicted_tags)}) for sentence {idx}.")
            y_true.extend(true_tags)
            y_pred.extend(predicted_tags)
            if idx % 100 == 0:
                logging.info(f"Processed {idx + 1} sentences in test data.")

        # Ensure y_true and y_pred have the same length
        if len(y_true) != len(y_pred):
            logging.error(f"Length mismatch after processing: True tags ({len(y_true)}) vs Predicted tags ({len(y_pred)}).")
            continue

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        f05 = 1.25 * precision * recall / (0.25 * precision + recall)
        f2 = 5 * precision * recall / (4 * precision + recall)

        overall_precision.append(precision)
        overall_recall.append(recall)
        overall_f1.append(f1)
        overall_f05.append(f05)
        overall_f2.append(f2)

        # Calculate per-tag metrics
        for tag in all_tags:
            true_tag_indices = [i for i, t in enumerate(y_true) if t == tag]
            pred_tag_indices = [i for i, t in enumerate(y_pred) if t == tag]

            if true_tag_indices and pred_tag_indices:
                p, r, f, _ = precision_recall_fscore_support(
                    [y_true[i] for i in true_tag_indices],
                    [y_pred[i] for i in true_tag_indices],
                    average='macro',
                    zero_division=0
                )
            else:
                p, r, f = 0, 0, 0

            per_tag_results[tag]['precision'].append(p)
            per_tag_results[tag]['recall'].append(r)
            per_tag_results[tag]['f1'].append(f)

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=all_tags)
        conf_matrix += cm

        logging.info(f"Completed fold {fold_number}.")
        fold_number += 1

    # Calculate average metrics
    avg_precision = np.mean(overall_precision)
    avg_recall = np.mean(overall_recall)
    avg_f1 = np.mean(overall_f1)
    avg_f05 = np.mean(overall_f05)
    avg_f2 = np.mean(overall_f2)

    # Save overall performance metrics
    overall_metrics = {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'f0.5': avg_f05,
        'f2': avg_f2
    }
    
    with open('overall_performance_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    logging.info("Overall performance metrics saved to 'overall_performance_metrics.json'.")

    # Save per-POS performance metrics
    per_tag_metrics = {
        tag: {
            'precision': np.mean(results['precision']),
            'recall': np.mean(results['recall']),
            'f1': np.mean(results['f1'])
        }
        for tag, results in per_tag_results.items()
    }
    
    with open('per_pos_performance_metrics.json', 'w') as f:
        json.dump(per_tag_metrics, f, indent=4)
    logging.info("Per-POS performance metrics saved to 'per_pos_performance_metrics.json'.")

    # Save confusion matrix
    np.save('confusion_matrix.npy', conf_matrix)
    logging.info("Confusion matrix saved to 'confusion_matrix.npy'.")

    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=all_tags, yticklabels=all_tags)  # Use '.2f' for two decimal float format
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')  # Save plot to file
    plt.show()
    logging.info("Confusion matrix plot saved as 'confusion_matrix.png'.")

    # Find most mismatched tags
    mismatches = []
    for i, tag1 in enumerate(all_tags):
        for j, tag2 in enumerate(all_tags):
            if i != j:
                mismatches.append((conf_matrix[i, j], tag1, tag2))

    mismatches.sort(reverse=True)
    most_mismatched_tags = mismatches[:5]  # Top 5 mismatches

    # Save most mismatched tags
    with open('most_mismatched_tags.json', 'w') as f:
        json.dump(most_mismatched_tags, f, indent=4)
    logging.info("Most mismatched tags saved to 'most_mismatched_tags.json'.")

if __name__ == "__main__":
    # Preprocess data
    tagged_sentences = preprocess_data()

    # Create and evaluate model
    hmm_tagger = HMMPOSTagger()
    evaluate_model(tagged_sentences, hmm_tagger, k=5)

    # Save the trained model
    hmm_tagger.save_model('hmm_pos_tagger.pkl')
