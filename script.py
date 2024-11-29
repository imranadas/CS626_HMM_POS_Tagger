import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_hmm_model(model_path='hmm_pos_tagger.pkl'):
    """Load the trained HMM model"""
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict

def create_transition_matrix_df(model_dict):
    """Create a DataFrame for the transition matrix with proper labels"""
    transition_matrix = np.exp(model_dict['transition_matrix'])  # Convert from log probabilities
    tags = list(model_dict['tag_to_idx'].keys())
    
    return pd.DataFrame(
        transition_matrix,
        index=tags,
        columns=tags
    )

def create_emission_matrix_df(model_dict):
    """Create a DataFrame for the emission matrix with proper labels"""
    emission_matrix = np.exp(model_dict['emission_matrix'])  # Convert from log probabilities
    tags = list(model_dict['tag_to_idx'].keys())
    words = list(model_dict['word_to_idx'].keys())
    
    return pd.DataFrame(
        emission_matrix,
        index=tags,
        columns=words
    )

def save_transition_heatmap(transition_df, filename='transition_matrix_heatmap.png'):
    """Save transition matrix heatmap"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(transition_df, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Transition Probabilities Between POS Tags')
    plt.ylabel('From Tag')
    plt.xlabel('To Tag')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def save_emission_heatmap(emission_df, n_top_words=20, filename='emission_matrix_heatmap.png'):
    """Save emission matrix heatmap for top words"""
    # Create condensed emission matrix with top words
    top_emissions = {}
    for tag in emission_df.index:
        top_words = emission_df.loc[tag].nlargest(n_top_words)
        top_emissions[tag] = top_words
    
    emission_condensed = pd.DataFrame(top_emissions).T
    
    plt.figure(figsize=(20, 10))
    sns.heatmap(emission_condensed, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title(f'Emission Probabilities (Top {n_top_words} Words per Tag)')
    plt.ylabel('POS Tag')
    plt.xlabel('Words')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def save_combined_heatmap(transition_df, emission_df, n_top_words=20, filename='combined_heatmap.png'):
    """Save both matrices in a single figure"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 24))
    
    # Plot transition matrix
    sns.heatmap(transition_df, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1)
    ax1.set_title('Transition Probabilities Between POS Tags')
    ax1.set_ylabel('From Tag')
    ax1.set_xlabel('To Tag')
    
    # Create condensed emission matrix with top words
    top_emissions = {}
    for tag in emission_df.index:
        top_words = emission_df.loc[tag].nlargest(n_top_words)
        top_emissions[tag] = top_words
    
    emission_condensed = pd.DataFrame(top_emissions).T
    
    # Plot condensed emission matrix
    sns.heatmap(emission_condensed, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax2)
    ax2.set_title(f'Emission Probabilities (Top {n_top_words} Words per Tag)')
    ax2.set_ylabel('POS Tag')
    ax2.set_xlabel('Words')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def display_matrix_statistics(transition_df, emission_df):
    """Display statistical information about both matrices and save to file"""
    stats = []
    
    stats.append("Transition Matrix Statistics:")
    stats.append("-" * 50)
    stats.append(f"Shape: {transition_df.shape}")
    stats.append("\nMost likely transitions (probability > 0.3):")
    
    high_trans = []
    for from_tag in transition_df.index:
        for to_tag in transition_df.columns:
            prob = transition_df.loc[from_tag, to_tag]
            if prob > 0.3:
                high_trans.append({
                    'From': from_tag,
                    'To': to_tag,
                    'Probability': prob
                })
    stats.append(pd.DataFrame(high_trans).sort_values('Probability', ascending=False).to_string())
    
    stats.append("\nEmission Matrix Statistics:")
    stats.append("-" * 50)
    stats.append(f"Shape: {emission_df.shape}")
    stats.append("\nMost characteristic words for each POS tag (highest probability):")
    
    top_words = {}
    for tag in emission_df.index:
        top_5 = emission_df.loc[tag].nlargest(5)
        top_words[tag] = [f"{word} ({prob:.3f})" for word, prob in top_5.items()]
    
    stats.append(pd.DataFrame(top_words, index=[f'Top {i+1}' for i in range(5)]).to_string())
    
    # Save statistics to file
    with open('matrix_statistics.txt', 'w') as f:
        f.write('\n'.join(str(stat) for stat in stats))
    
    # Print to console as well
    for stat in stats:
        print(stat)

def main():
    # Load the model
    print("Loading model...")
    model_dict = load_hmm_model()
    
    # Create matrices DataFrames
    print("Processing matrices...")
    transition_df = create_transition_matrix_df(model_dict)
    emission_df = create_emission_matrix_df(model_dict)
    
    # Save individual heatmaps
    print("Saving visualizations...")
    transition_file = save_transition_heatmap(transition_df)
    emission_file = save_emission_heatmap(emission_df)
    combined_file = save_combined_heatmap(transition_df, emission_df)
    
    # Display and save statistics
    print("\nGenerating statistics...")
    display_matrix_statistics(transition_df, emission_df)
    
    # Save full matrices to CSV
    print("\nSaving matrices to CSV files...")
    transition_df.to_csv('transition_matrix.csv')
    emission_df.to_csv('emission_matrix.csv')
    
    print("\nFiles saved:")
    print(f"- {transition_file}")
    print(f"- {emission_file}")
    print(f"- {combined_file}")
    print("- matrix_statistics.txt")
    print("- transition_matrix.csv")
    print("- emission_matrix.csv")

if __name__ == "__main__":
    main()