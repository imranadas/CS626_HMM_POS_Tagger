import streamlit as st
from hmm_training import HMMPOSTagger
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
from collections import defaultdict

# Page configuration
st.set_page_config(
    page_title="HMM POS Tagger",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache decorators for resource loading
@st.cache_resource
def download_nltk_resources():
    resources = ['punkt', 'averaged_perceptron_tagger', 'universal_tagset']
    for resource in resources:
        nltk.download(resource)

@st.cache_resource
def load_tagger(model_path='hmm_pos_tagger.pkl'):
    """Load the trained HMM tagger model"""
    try:
        tagger = HMMPOSTagger.load_model(model_path)
        return tagger, None
    except Exception as e:
        return None, str(e)

def load_metrics_files():
    """Load all evaluation metrics files"""
    metrics = {}
    try:
        # Load overall metrics
        with open('overall_performance_metrics.json', 'r') as f:
            metrics['overall'] = json.load(f)
        
        # Load per-POS metrics
        with open('per_pos_performance_metrics.json', 'r') as f:
            metrics['per_pos'] = json.load(f)
        
        # Load mismatched tags
        with open('most_mismatched_tags.json', 'r') as f:
            metrics['mismatches'] = json.load(f)
        
        # Load confusion matrix
        metrics['confusion_matrix'] = np.load('confusion_matrix.npy')
        
        return metrics, None
    except Exception as e:
        return None, str(e)
    
def validate_sentence(tokens, tags):
    """Validate that tokens and tags match in length"""
    if len(tokens) != len(tags):
        return False, f"Mismatch: {len(tokens)} tokens vs {len(tags)} tags"
    return True, None

def evaluate_test_data(tagger, test_data):
    """Evaluate model performance on the provided test data"""
    results = []
    total_correct = 0
    total_tags = 0
    validation_errors = []
    
    for i, item in enumerate(test_data):
        # Tokenize sentence
        tokens = word_tokenize(item['sentence'].lower())
        true_tags = item['tags']
        
        # Validate lengths
        is_valid, error_msg = validate_sentence(tokens, true_tags)
        if not is_valid:
            validation_errors.append({
                'index': i,
                'sentence': item['sentence'],
                'error': error_msg,
                'tokens': tokens,
                'tags': true_tags
            })
            continue
            
        # Get model predictions
        predicted_tags = tagger.viterbi_improved(tokens)
        
        # Calculate accuracy for this sentence
        correct_tags = sum(1 for pred, true in zip(predicted_tags, true_tags) if pred == true)
        sentence_accuracy = correct_tags / len(true_tags)
        
        # Update totals
        total_correct += correct_tags
        total_tags += len(true_tags)
        
        # Store detailed results
        results.append({
            'sentence': item['sentence'],
            'tokens': tokens,
            'true_tags': true_tags,
            'predicted_tags': predicted_tags,
            'accuracy': sentence_accuracy
        })
    
    overall_accuracy = total_correct / total_tags if total_tags > 0 else 0
    return results, overall_accuracy, validation_errors

def create_tag_comparison_table(tokens, true_tags, predicted_tags):
    """Create a formatted comparison table for tags"""
    df = pd.DataFrame({
        'Token': tokens,
        'True Tag': true_tags,
        'Predicted Tag': predicted_tags
    })
    
    # Create background colors based on tag matches
    def highlight_matches(row):
        color = '#90EE90' if row['True Tag'] == row['Predicted Tag'] else '#FFB6C6'
        return [''] * 2 + [f'background-color: {color}']
    
    return df.style.apply(highlight_matches, axis=1)

def create_evaluation_charts(results):
    """Create visualizations for test data evaluation results"""
    if not results:
        return None, None
        
    # Sentence-level accuracy distribution
    accuracies = [r['accuracy'] for r in results]
    acc_hist = px.histogram(
        x=accuracies,
        nbins=20,
        title="Distribution of Sentence-level Accuracies",
        labels={'x': 'Accuracy', 'y': 'Count'},
        height=400
    )
    
    # Tag confusion analysis
    tag_pairs = []
    for result in results:
        for true, pred in zip(result['true_tags'], result['predicted_tags']):
            tag_pairs.append({'True': true, 'Predicted': pred})
    
    if tag_pairs:
        confusion_df = pd.DataFrame(tag_pairs)
        confusion_matrix = pd.crosstab(confusion_df['True'], confusion_df['Predicted'])
        
        confusion_fig = px.imshow(
            confusion_matrix,
            title="Tag Confusion Matrix",
            labels=dict(x="Predicted Tag", y="True Tag", color="Count"),
            height=500
        )
    else:
        confusion_fig = None
    
    return acc_hist, confusion_fig

def create_performance_charts(metrics):
    """Create all performance visualization charts"""
    # Overall performance bar chart
    overall_fig = go.Figure()
    for metric, value in metrics['overall'].items():
        overall_fig.add_trace(go.Bar(
            name=metric,
            x=[metric.capitalize()],
            y=[value * 100],
            text=[f"{value * 100:.2f}%"],
            textposition='auto',
        ))
    overall_fig.update_layout(
        title="Overall Model Performance",
        yaxis_title="Percentage",
        showlegend=False,
        height=400
    )

    # Per-POS metrics chart
    pos_data = []
    for tag, tag_metrics in metrics['per_pos'].items():
        for metric, value in tag_metrics.items():
            pos_data.append({
                'POS Tag': tag,
                'Metric': metric.capitalize(),
                'Value': value * 100
            })
    pos_fig = px.bar(
        pd.DataFrame(pos_data),
        x='POS Tag',
        y='Value',
        color='Metric',
        barmode='group',
        title='Per POS Tag Performance',
        labels={'Value': 'Percentage'},
        height=500
    )

    # Mismatches heatmap
    mismatches_df = pd.DataFrame(metrics['mismatches'], 
                                columns=['Count', 'True Tag', 'Predicted Tag'])
    mismatches_fig = px.density_heatmap(
        mismatches_df,
        x='Predicted Tag',
        y='True Tag',
        z='Count',
        title='Most Frequent Tag Mismatches',
        labels={'Count': 'Frequency'},
        height=400
    )

    # Confusion matrix
    cm_fig = px.imshow(
        metrics['confusion_matrix'],
        labels=dict(x="Predicted Tag", y="True Tag", color="Count"),
        title="Confusion Matrix"
    )

    return overall_fig, pos_fig, mismatches_fig, cm_fig

def create_tag_table(tagged_df):
    """Create an interactive table visualization of tagged results using Streamlit's color scheme"""
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Word', 'POS Tag', 'Explanation'],
            fill_color='#0e1117',  # Streamlit's dark background
            font=dict(
                size=14,
                color='white'
            ),
            align='left',
            line_color='#1e1e1e'  # Subtle border color
        ),
        cells=dict(
            values=[
                tagged_df['Word'],
                tagged_df['POS Tag'],
                tagged_df['Explanation']
            ],
            fill_color=['#1e1e1e'],  # Slightly lighter than background
            font=dict(
                size=12,
                color='#fafafa'  # Light text for contrast
            ),
            align='left',
            line_color='#2d2d2d'  # Subtle border color
        )
    )])
    
    # Update layout to match Streamlit's container style
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#0e1117',  # Match Streamlit's background
        plot_bgcolor='#0e1117',
        height=400,  # Fixed height for better integration
        font=dict(
            family="'Source Sans Pro', -apple-system, 'system-ui', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', sans-serif"
        )
    )
    
    return fig

def get_pos_explanations():
    """Get detailed explanations for POS tags"""
    return {
        'NOUN': 'Nouns (people, places, things, concepts)',
        'VERB': 'Verbs (actions, states, occurrences)',
        'ADJ': 'Adjectives (describe nouns)',
        'ADV': 'Adverbs (modify verbs, adjectives, other adverbs)',
        'PRON': 'Pronouns (replace nouns)',
        'DET': 'Determiners (articles, demonstratives)',
        'ADP': 'Adpositions (prepositions, postpositions)',
        'NUM': 'Numerals (numbers, quantities)',
        'CONJ': 'Conjunctions (connect words/phrases)',
        'PRT': 'Particles (function words, miscellaneous)',
        '.': 'Punctuation marks',
        'X': 'Other or unknown categories'
    }

def tag_text(tagger, text, explanations):
    """Process text and return tagged DataFrame with explanations"""
    tokens = word_tokenize(text.lower())
    tags = tagger.viterbi_improved(tokens)
    
    return pd.DataFrame({
        'Word': tokens,
        'POS Tag': tags,
        'Explanation': [explanations.get(tag, '') for tag in tags]
    })

def main():
    """Main application function"""
    st.title("HMM Part-of-Speech Tagger")
    
    # Download NLTK resources
    with st.spinner("Loading resources..."):
        download_nltk_resources()

    # Load model and metrics
    tagger, tagger_error = load_tagger()
    metrics, metrics_error = load_metrics_files()
    
    if tagger_error:
        st.error(f"Error loading model: {tagger_error}")
        return
    if metrics_error:
        st.error(f"Error loading metrics: {metrics_error}")
        return

    # Main navigation
    tab1, tab2, tab3, tab4 = st.tabs(["Tag Text", "Model Performance", "Test Evaluation", "About"])
    
    # Tab 1: Text Tagging
    with tab1:
        st.header("Tag Text")
        text_input = st.text_area(
            "Enter text to tag",
            value="The quick brown fox jumps over the lazy dog.",
            height=100
        )
        
        col1, col2 = st.columns([1, 2])
        if st.button("Tag Text"):
            if text_input.strip():
                with st.spinner("Processing text..."):
                    # Tag text
                    explanations = get_pos_explanations()
                    results_df = tag_text(tagger, text_input, explanations)
                    
                    # Display results
                    st.subheader("Tagged Results")
                    fig = create_tag_table(results_df)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add download button
                    st.download_button(
                        "Download Results (CSV)",
                        results_df.to_csv(index=False),
                        "tagged_text.csv",
                        "text/csv"
                    )
            else:
                st.warning("Please enter some text to tag.")

    # Tab 2: Model Performance
    with tab2:
        st.header("Model Performance Analysis")
        
        # Create performance visualizations
        overall_fig, pos_fig, mismatches_fig, cm_fig = create_performance_charts(metrics)
        
        # Display charts in organized layout
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(overall_fig, use_container_width=True)
        with col2:
            st.plotly_chart(mismatches_fig, use_container_width=True)
        
        st.plotly_chart(pos_fig, use_container_width=True)
        st.plotly_chart(cm_fig, use_container_width=True)
        
        # Add metrics download section
        st.subheader("Download Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Overall Metrics (JSON)",
                json.dumps(metrics['overall'], indent=2),
                "overall_metrics.json",
                "application/json"
            )
        with col2:
            st.download_button(
                "Per-POS Metrics (JSON)",
                json.dumps(metrics['per_pos'], indent=2),
                "pos_metrics.json",
                "application/json"
            )
        with col3:
            st.download_button(
                "Tag Mismatches (JSON)",
                json.dumps(metrics['mismatches'], indent=2),
                "mismatches.json",
                "application/json"
            )

    with tab3:
        st.header("Test Data Evaluation")
        
        # Load and evaluate test data
        try:
            with open('test_data.json', 'r') as f:
                test_data = json.load(f)
            
            st.info(f"Loaded {len(test_data)} test sentences")
            
            if st.button("Run Evaluation"):
                with st.spinner("Evaluating test data..."):
                    results, overall_accuracy, validation_errors = evaluate_test_data(tagger, test_data)
                    
                    # Display validation errors if any
                    if validation_errors:
                        st.error(f"Found {len(validation_errors)} sentences with validation errors")
                        with st.expander("View Validation Errors"):
                            for error in validation_errors:
                                st.markdown(f"""
                                **Sentence {error['index'] + 1}**: {error['sentence']}
                                - {error['error']}
                                - Tokens ({len(error['tokens'])}): `{error['tokens']}`
                                - Tags ({len(error['tags'])}): `{error['tags']}`
                                ---
                                """)
                    
                    if results:
                        # Display overall accuracy
                        st.metric(
                            "Overall Accuracy",
                            f"{overall_accuracy:.2%}",
                            help="Percentage of correctly tagged words across valid test sentences"
                        )
                        
                        # Create and display charts
                        acc_hist, confusion_fig = create_evaluation_charts(results)
                        if acc_hist:
                            st.plotly_chart(acc_hist, use_container_width=True)
                        if confusion_fig:
                            st.plotly_chart(confusion_fig, use_container_width=True)
                        
                        # Display detailed results in an expandable section
                        with st.expander("Detailed Results"):
                            for i, result in enumerate(results, 1):
                                st.markdown(f"**Sentence {i}** (Accuracy: {result['accuracy']:.2%})")
                                st.write("Text:", result['sentence'])
                                
                                # Create and display comparison table with fixed styling
                                comparison_table = create_tag_comparison_table(
                                    result['tokens'],
                                    result['true_tags'],
                                    result['predicted_tags']
                                )
                                st.dataframe(comparison_table)
                                st.markdown("---")
                        
                        # Add summary statistics
                        st.subheader("Tag-wise Performance")
                        tag_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
                        for result in results:
                            for true_tag, pred_tag in zip(result['true_tags'], result['predicted_tags']):
                                tag_stats[true_tag]['total'] += 1
                                if true_tag == pred_tag:
                                    tag_stats[true_tag]['correct'] += 1
                        
                        tag_performance = []
                        for tag, stats in tag_stats.items():
                            accuracy = stats['correct'] / stats['total']
                            tag_performance.append({
                                'Tag': tag,
                                'Accuracy': f"{accuracy:.2%}",
                                'Correct/Total': f"{stats['correct']}/{stats['total']}"
                            })
                        
                        st.table(pd.DataFrame(tag_performance).sort_values('Tag'))
                        
                        # Add download button for detailed results
                        results_df = pd.DataFrame([{
                            'Sentence': r['sentence'],
                            'Accuracy': r['accuracy'],
                            'True Tags': ' '.join(r['true_tags']),
                            'Predicted Tags': ' '.join(r['predicted_tags'])
                        } for r in results])
                        
                        st.download_button(
                            "Download Detailed Results (CSV)",
                            results_df.to_csv(index=False),
                            "test_evaluation_results.csv",
                            "text/csv"
                        )
                        
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")
            st.exception(e)
    
    # Tab 3: About
    with tab4:
        st.header("About the Model")
        st.markdown("""
        ### HMM POS Tagger
        This is an enhanced Hidden Markov Model (HMM) for Part-of-Speech tagging, 
        trained on the Brown corpus using the Universal Dependencies tagset.
        
        #### Key Features
        - Morphological analysis for unknown words
        - Enhanced emission probability calculation
        - Improved rare word handling
        - Feature-based tagging
        - Laplace smoothing
        
        #### Universal POS Tags
        """)
        
        # Display POS tag explanations in a clean format
        explanations = get_pos_explanations()
        for tag, explanation in explanations.items():
            st.markdown(f"**{tag}**: {explanation}")
        
        st.markdown("""
        #### Usage Tips
        - Enter any English text in the "Tag Text" tab
        - Choose different unknown word handling methods
        - View detailed performance metrics in the "Model Performance" tab
        - Download results and metrics for further analysis
        
        #### Model Performance
        The model achieves high accuracy through:
        - Sophisticated unknown word handling
        - Context-aware tagging
        - Robust probability estimation
        """)

if __name__ == "__main__":
    main()