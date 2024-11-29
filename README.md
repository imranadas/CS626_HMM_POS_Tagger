# Enhanced HMM POS Tagger

This project implements an enhanced Hidden Markov Model (HMM) for Part-of-Speech (POS) tagging using Python. It features advanced unknown word handling, morphological analysis, and a comprehensive Streamlit web application for model interaction and performance visualization.

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Web Application](#web-application)
6. [Model Details](#model-details)

## Features

- **Advanced POS Tagging**:
  - Morphological analysis for unknown words
  - Enhanced emission probability calculation
  - Feature-based tagging system
  - Improved rare word handling
  - Laplace smoothing

- **Comprehensive Web Interface**:
  - Interactive text tagging
  - Detailed performance visualizations
  - Multiple unknown word handling methods
  - Downloadable results and metrics

- **Performance Analysis**:
  - Confusion matrix visualization
  - Per-POS tag performance metrics
  - Most common tag mismatches
  - Overall model evaluation metrics

## Prerequisites

- Python 3.7 or later
- pip (Python package installer)

Required Python packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install nltk numpy scikit-learn matplotlib seaborn pandas streamlit plotly
```

## Installation

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/enhanced-hmm-pos-tagger.git
cd enhanced-hmm-pos-tagger
```

2. **Download NLTK Resources**
```python
import nltk
nltk.download(['brown', 'universal_tagset', 'punkt'])
```

## Usage

1. **Train the Model**
```bash
python improved_training.py
```

This generates:
- Trained model (`hmm_pos_tagger.pkl`)
- Performance metrics files:
  - `confusion_matrix.npy`
  - `confusion_matrix.png`
  - `most_mismatched_tags.json`
  - `overall_performance_metrics.json`
  - `per_pos_performance_metrics.json`

2. **Launch the Web Application**
```bash
streamlit run improved_app.py
```

## Web Application

The Streamlit application offers three main sections:

### 1. Tag Text
- Input any English text
- Choose unknown word handling method:
  - Morphological Analysis
  - Feature-based Analysis
  - Most Common Tag
- View POS tags with explanations
- Download tagged results as CSV

### 2. Model Performance
- Interactive visualizations:
  - Overall performance metrics
  - Per-POS tag performance
  - Confusion matrix
  - Tag mismatch patterns
- Download performance metrics

### 3. About
- Model information
- POS tag explanations
- Usage guidelines
- Performance details

## Model Details

### Training Data
- Brown Corpus (NLTK)
- Universal Dependencies POS tagset

### Unknown Word Handling
1. **Morphological Analysis**:
   - Suffix/prefix patterns
   - Word structure analysis
   - Common linguistic patterns

2. **Feature-based Analysis**:
   - Capitalization
   - Numerical content
   - Special characters
   - Common affixes

3. **Statistical Backup**:
   - Emission probabilities
   - Transition patterns
   - Laplace smoothing

### POS Tags
- NOUN: Nouns
- VERB: Verbs
- ADJ: Adjectives
- ADV: Adverbs
- PRON: Pronouns
- DET: Determiners
- ADP: Adpositions
- NUM: Numerals
- CONJ: Conjunctions
- PRT: Particles
- X: Other categories
- .: Punctuation

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Verify all files are present in the correct directory
   - Check file permissions
   - Ensure consistent Python versions

2. **Memory Issues**
   - Reduce batch size during training
   - Close other memory-intensive applications
   - Consider using a machine with more RAM

3. **Visualization Errors**
   - Update Plotly and Streamlit to latest versions
   - Clear browser cache
   - Check console for JavaScript errors

### Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Provide clear reproduction steps

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK Project for the Brown Corpus
- Streamlit team for the web framework
- Universal Dependencies project for the POS tagset
