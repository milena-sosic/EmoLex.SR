import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from os.path import sep, join
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from constants import *

model = SentenceTransformer(SBERT_MODEL_PATH)

def pjoin(*args, **kwargs):
    return join(*args, **kwargs).replace(sep, '/')


def similarity_filter(sentences):
    """
    Recursively check if sentences pass a similarity filter.
    :param sentences: list of strings, contains sentences.
    If the function finds sentences that fail the similarity test, the above param will be the function output.
    :return: this method upon itself unless there are no similar sentences; in that case the feed that was passed
    in is returned.
    """

    # Remove similar sentences
    all_summary_pairs = list(combinations(sentences, 2))
    similar_sentences = []
    for pair in all_summary_pairs:
        sent1_embedding = model.encode([pair[0]]).reshape(1, -1)
        sent2_embedding = model.encode([pair[1]]).reshape(1, -1)
        similarity = cosine_similarity(sent1_embedding, sent2_embedding)[0][0]
        # print(similarity)
        if similarity > 0.75:
            print(pair)
            similar_sentences.append(pair)

    sentences_to_remove = []
    for a_sentence in similar_sentences:
        # Get the index of the first sentence in the pair
        index_for_removal = sentences.index(a_sentence[0])
        sentences_to_remove.append(index_for_removal)

    # Get indices of similar sentences and remove them
    similar_sentence_counts = set(sentences_to_remove)
    similar_sentences = [
        x[1] for x in enumerate(sentences) if x[0] in similar_sentence_counts
    ]

    # Exit the recursion if there are no longer any similar sentences
    if len(similar_sentence_counts) == 0:
        return sentences

    # Continue the recursion if there are still sentences to remove
    else:
        # Remove similar sentences from the next input
        for sentence in similar_sentences:
            idx = sentences.index(sentence)
            sentences.pop(idx)

        return similarity_filter(sentences)
    

def process_llm_sentences(file_path, output_path):
    """
    Processes LLM sentences for validation and correction.
    """
    llm_sent_df = pd.read_csv(file_path, sep=',')
    
    # Calculate and print statistics
    total_rows = llm_sent_df.shape[0]
    incorrect_sentences = 100 * llm_sent_df[~llm_sent_df['sentence_sr_corr'].isna()].shape[0] / total_rows
    incorrect_labels = 100 * llm_sent_df[~llm_sent_df['label_corr'].isna()].shape[0] / total_rows
    incorrect_total = 100 * llm_sent_df[(~llm_sent_df['label_corr'].isna()) | (~llm_sent_df['sentence_sr_corr'].isna())].shape[0] / total_rows
    
    print(f'Incorrect sentences: {incorrect_sentences:.2f}%')
    print(f'Incorrect labels: {incorrect_labels:.2f}%')
    print(f'Incorrect Total: {incorrect_total:.2f}%')

    # Filter out rows marked for dropping
    llm_sent_df = llm_sent_df[llm_sent_df['Action'] != 'drop']

    # Apply corrections
    llm_sent_df['sentence_sr'] = np.where(~llm_sent_df['sentence_sr'].isna(), llm_sent_df['sentence_sr'], llm_sent_df['sentence_sr'])
    llm_sent_df['label'] = np.where(~llm_sent_df['label_corr'].isna(), llm_sent_df['label_corr'], llm_sent_df['label'])

    # Filter labels
    def filter_labels(label):
        return [y.strip() for y in label.strip('[]').split(',') if y.strip() not in EMO_CATEGORIES]

    llm_sent_df['label'] = llm_sent_df['label'].apply(filter_labels)
    llm_sent_df = llm_sent_df[llm_sent_df['label'].str.len() != 0].reset_index(drop=True)

    # Save the processed DataFrame
    llm_sent_df[['sentence_en', 'sentence_sr', 'Label']].to_csv(output_path, sep='\t', index=False)
    return llm_sent_df


def main():
    data_path = os.getcwd() + '/data'
    df = process_llm_sentences(pjoin(data_path, 'LLM-Emo.SR-orig-manual.csv'), pjoin(data_path, 'LLM-Emo.SR.val.csv'))

    unique_sentences = similarity_filter(df.sentence_sr.values.tolist())
    print(len(unique_sentences))

    df_us = pd.DataFrame(unique_sentences, columns=['sentence_sr'])
    df_us['unique'] = True
    df = df.merge(df_us, on='sentence_sr', how='left')
    
    df = df[df['unique'] == True]
    print("Unique sentences: ", df.shape[0])
    file_path = pjoin(data_path, 'LLM-Emo.SR.csv')
    df[['sentence_en', 'sentence_sr', 'label']].to_csv(file_path, sep='\t')


if __name__ == '__main__':
    main()
