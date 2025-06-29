import pandas as pd
import numpy as np
from constants import *
import nltk
import re
from string import punctuation
from nltk.stem import wordnet
import cyrtranslit
import math
from os.path import sep, join
from typing import List
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from itertools import chain
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.model_selection import train_test_split as sk_train_test_split


def pjoin(*args, **kwargs):
    return join(*args, **kwargs).replace(sep, '/')


def extract_wna_emo(wna_sr, nlp, text):
    tokens = nltk.wordpunct_tokenize(text)
    tokens_df = pd.DataFrame(tokens, columns=['word_sr'])
    
    tokens_df = pd.merge(tokens_df, nlp, on=['word_sr'], how='left')
    tokens_df = tokens_df[['word_sr', 'pos_sr', 'lemma_sr']]
    tokens_df = pd.merge(tokens_df, wna_sr, on=['lemma_sr', 'pos_sr'], how='left')
    a = tokens_df['wna_label'].fillna('').str.split(',').tolist()

    a = list(set([i for j in a for i in j]))
    a = [i.replace(" ", "") for i in a if i != '']
    b = [k for k, v in NRC_WNA_MAPPING.items() for i in a if i in v]
    
    return '|'.join(list(set(b)))


def affect_label(df, labelList: List[str]):
    """
    Determines the harmonized label based on multiple label columns.
    """
    ann = df.copy()
    mlb = MultiLabelBinarizer()

    def process_label(label_column):
        """Processes a label column into a binary matrix."""
        labels = ann[label_column].fillna('').str.split(',').apply(lambda x: [y.strip() for y in x])
        return pd.DataFrame(mlb.fit_transform(labels), columns=mlb.classes_, index=ann.index)

    combined_df = pd.DataFrame()
    for label in labelList:
        ann = process_label(label)
        combined_df = combined_df.add(ann, fill_value=0)

    # Determine the maximum and second maximum labels
    is_max = combined_df.eq(combined_df.max(axis=1), axis=0)
    df1 = combined_df.mask(is_max, combined_df.min(axis=1), axis=0)
    is_max_second = df1.eq(df1.max(axis=1), axis=0) & df1.gt(0, axis=0)
    is_max_final = np.logical_or(is_max, is_max_second)

    # Create the result column
    result = is_max_final.dot(combined_df.columns + " ").str.split(' ').apply(
        lambda x: ','.join([y.strip() for y in x if y != '']))

    return result


def translate_word_en_to_sr_google(translator, text):
    txt = text
    output = []
    if len(txt) > 5000:
        while len(txt) > 0:
            output.append(translator.translate(txt[:4999]))
            txt = txt[4999:]
    else:
        output.append(translator.translate(txt))
    text_tr = ' '.join(output) if output[0] is not None else ''
    text_lat = cyrtranslit.to_latin(text_tr)
    return text_lat


def reflexive_verb(word_sr_lat):
    parts = word_sr_lat.strip().split(' ')
    reflexive = ['sebe', 'se']
    ref = 0
    rt_val = word_sr_lat
    if len(parts) == 2:
        if parts[1] in reflexive:
            rt_val = [parts[0]]
            ref = 1
        else:
            rt_val = parts
    else:
        rt_val = parts
    rt_val = ' '.join(rt_val)
    return word_sr_lat, rt_val, ref


def convert_pos_tag(penn_tag):
    """
    Convert a POS tag from the Penn Treebank tagset to the Universal Dependencies (UD) tagset.
    """
    mapping = {
        'NN': 'NOUN',  
        'NNS': 'NOUN',  
        'NNP': 'PROPN', 
        'NNPS': 'PROPN',
        'VB': 'VERB',   
        'VBD': 'VERB',  
        'VBG': 'VERB',  
        'VBN': 'VERB',  
        'VBP': 'VERB',  
        'VBZ': 'VERB',  
        'JJ': 'ADJ',    
        'JJR': 'ADJ',   
        'JJS': 'ADJ',   
        'RB': 'ADV',    
        'RBR': 'ADV',   
        'RBS': 'ADV',   
        'IN': 'ADP',    
        'DT': 'DET',    
        'PRP': 'PRON',  
        'PRP$': 'PRON', 
        'CC': 'CCONJ',  
        'CD': 'NUM',    
        'UH': 'INTJ',   
        'FW': 'X',      
    }
    # Default to 'X' if the tag is not found
    return mapping.get(penn_tag, 'X')


def nltk_tag_to_wordnet_tag(nltk_tag):
    # print(nltk_tag)
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def convert_wordnet_tag(nltk_tag):
    # print(nltk_tag)
    if nltk_tag.startswith('a'):
        return 'ADJ'
    elif nltk_tag.startswith('v'):
        return 'VERB'
    elif nltk_tag.startswith('n'):
        return 'NOUN'
    elif nltk_tag.startswith('r'):
        return 'ADV'
    elif nltk_tag.startswith('b'):
        return 'ADV'
    elif nltk_tag.startswith('X'):
        return 'X'
    else:
        return None


def multilabel_train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    if stratify is None:
        
        return sk_train_test_split(*arrays, test_size=test_size, train_size=train_size,
                                    random_state=random_state, stratify=None, shuffle=shuffle)

    assert shuffle, "Stratified train/test split is not implemented for shuffle=False"
    n_arrays = len(arrays)
    arrays = indexable(*arrays)
    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size, default_test_size=0.25)
    cv = MultilabelStratifiedShuffleSplit(test_size=n_test, train_size=n_train, random_state=123)
    train, test = next(cv.split(X=arrays[0], y=stratify))

    return list(chain.from_iterable((_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays))


def normal_round(n):
    """Rounds a number to the nearest integer."""
    return math.floor(n) if n - math.floor(n) < 0.5 else math.ceil(n)


def remove_hashtag_usernames_links(tweet, mask=True):
    if mask:
        tweet = re.sub(r'#[^\s+]+', '#hashtag', tweet)
        tweet = re.sub(r'@[^\s+]+', '#user', tweet)
        tweet = re.sub(r'http[^\s+]+', '#masklink', tweet)
    else:
        tweet = re.sub(r'#[^\s+]+', '', tweet)
        tweet = re.sub(r'@[^\s+]+', '', tweet)
        tweet = re.sub(r'http[^\s+]+', '', tweet)
    return tweet


def restore_contractions(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def normalize_text(sentences):
    sent = sentences
    sent = cyrtranslit.to_latin(sent)
    sent = sent.lower()  
    sent = remove_hashtag_usernames_links(sent, mask=False)
    
    sent = restore_contractions(sent)
    sent = re.sub(r'([0-9]+)', '', sent)  
    punctuations = punctuation
    pattern = r"[{}]".format(punctuations)  
    sent = re.sub(pattern, ' ', sent)
    sent = re.sub(r'\s+', ' ', sent)
        
    return sent.strip()

def lemmatize_sentence_en(sentence, lemmatizer):
    # tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.wordpunct_tokenize(sentence))
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append({'word': word, 'pos': np.nan, 'lemma': np.nan})
        else:
            # else use the tag to lemmatize the token
            word_lemma = lemmatizer.lemmatize(word, tag)
            lemmatized_sentence.append({'word': word, 'pos': convert_wordnet_tag(tag), 'lemma': word_lemma})
    return lemmatized_sentence


def lemmatize_sentence_sr(sentence, nlp):
    tokens = [x for x in nltk.wordpunct_tokenize(sentence.lower()) if x not in HELPER_VERBS_SR]
    tokens_df = pd.DataFrame(tokens, columns=['word'])
    tokens_df = pd.merge(tokens_df, nlp, on=['word'], how='left')
    tokens_df = tokens_df[['word', 'pos', 'lemma']]
    tokens_df.fillna('', inplace=True)
    return ' '.join(tokens_df.lemma.tolist())

def clear_emo_labels(x):
    labels = x.strip(']').strip('[').split(',')
    labels = [x.strip(' ') for x in labels]
    labels = sorted(list(set([x for x in labels if x in EMO_CATEGORIES])))
    return ','.join(labels)
