import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_lexicons import load_sr_lexicon
from constants import MORHO_NEG_PREFIXES

def preprocess_data(df):
    """
    Preprocess the data: handle missing values and standardize text.
    """
    df.fillna('', inplace=True)
    df['lemma'] = df['lemma'].str.lower().str.strip()
    return df


def compare_labels(df):
    """
    Compare labels for negated and non-negated words.
    """
    comparison_results = []
   
    word_to_labels = df.set_index('lemma')['label'].apply(lambda x: set(x.split(','))).to_dict()

    for word in df['lemma']:
        for prefix in MORHO_NEG_PREFIXES:
            if word.startswith(prefix):
                non_negated_word = word[len(prefix):]
                negated_labels = word_to_labels.get(word, set())
                if non_negated_word in word_to_labels:
                    non_negated_labels = word_to_labels.get(non_negated_word, set())

                    non_negated_categories = {cat for label in non_negated_labels for cat in label.split('|')}
                    negated_categories = {cat for label in negated_labels for cat in label.split('|')}

                    comparison_results.append({
                        'word': non_negated_word,
                        'negated_word': word,
                        'labels': non_negated_categories,
                        'negated_labels': negated_categories,
                        'label_difference': non_negated_categories.symmetric_difference(negated_categories)
                    })
                break  
    return pd.DataFrame(comparison_results)


def calculate_transition(df):
    """
    Calculate a crosstab for emotion transitions.
    """
    transitions = []
    for _, row in df.iterrows():
        for label in row['labels']:
            for neg_label in row['negated_labels']:
                transitions.append((label, neg_label))

    transitions_df = pd.DataFrame(transitions, columns=['Original', 'Negated'])

    crosstab = pd.crosstab(transitions_df['Original'], transitions_df['Negated'], normalize='index')
    return crosstab


def get_top_opposite_emotions(transition_matrix, top_n=3):
    """
    Get the top opposite emotions based on the transition matrix for each emotion category.
    """
    top_opposites = {}

    for emotion, transitions in transition_matrix.items():
        # Sort the transitions based on probability in descending order
        sorted_transitions = sorted(transitions.items(), key=lambda item: item[1], reverse=True)
        
        # Select the top_n opposite emotions
        top_opposites[emotion] = [emotion for emotion, _ in sorted_transitions[:top_n]]

    return top_opposites


def analyze_negation_effects():
    """
    Load, preprocess, and analyze the negation effects in the EmoLex.SR-v2 lexicon.
    """
    df = load_sr_lexicon(version=2)
    df = preprocess_data(df)
    
    comparison_df = compare_labels(df)
    comparison_df.to_csv('./results/morphological_negations.csv', sep='\t', index=False)
    
    print("Comparison Results:")
    print(comparison_df)

    transition_crosstab = calculate_transition(comparison_df)
    print("Transition Crosstab (Percentage):")
    print(transition_crosstab)

    top_opposite_emotions = get_top_opposite_emotions(transition_crosstab.to_dict(orient='index'), top_n=1)

    for emotion, opposites in top_opposite_emotions.items():
        print(f"Top opposite emotions for {emotion}: {', '.join(opposites)}")


if __name__ == "__main__":
    analyze_negation_effects()
