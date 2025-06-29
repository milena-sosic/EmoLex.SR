import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from utils import *
from lexicon_validation.metrics import *
from sklearn.preprocessing import MultiLabelBinarizer
import cyrtranslit
from load_lexicons import *
from constants import *


def main():
    df = pd.read_csv(EMOLEX_SR_VAL_PATH, sep=',')
    df['lemma_sr_final'] = np.where(df['valid_word'] == 'word_sr_tr', df['word_sr_tr'],
                                    np.where(df['valid_word'] == 'correct_word_sr', df['correct_word_sr'],
                                             df['word_sr']))

    _, df['lemma_sr_final'], _ = zip(
        *df['lemma_sr_final'].fillna('').apply(lambda x: reflexive_verb(cyrtranslit.to_latin(x).lower())))

    df['label_sr_final'] = np.where(df['valid_label'] == 'label_gpt', df['label_gpt'],
                                    np.where(df['valid_label'] == 'label_ann', df['label_ann'],
                                             df['label_en']))

    df = df[df.lemma_sr_final != '']
    df = df[~df.label_sr_final.isna()]

    measure_accuracy(df, type='label')

    emo_categories = EMO_CATEGORIES + ['neutral']

    s1 = df['label_sr_final'].fillna('').str.split('|').apply(
        lambda x: [y.strip() for y in x if y in emo_categories])
    
    mlb = MultiLabelBinarizer()
    mlb_s1 = mlb.fit_transform(s1)

    ann1 = pd.DataFrame(mlb_s1, columns=mlb.classes_, index=df.index)
    df = pd.concat([df[['lemma_sr_final', 'pos_sr', 'synonyms_gpt', 'synonyms_swn', 'synonyms_incorrect', 'synonyms_manual', 'label_sr_final']], ann1], axis=1)

    df[emo_categories] = df[emo_categories].div(df[emo_categories].sum(axis=1), axis=0)

    synonym_agg_funcs = {synonym: lambda x: ','.join(filter(lambda x: x is not np.nan, list(x))) for synonym in SYNONYM_GROUPS}

    emotion_agg_funcs = {emotion: 'mean' for emotion in emo_categories}

    agg_funcs = {**synonym_agg_funcs, **emotion_agg_funcs}

    # Perform the aggregation
    df_agg = df.groupby(['lemma_sr_final', 'pos_sr']).agg(agg_funcs).reset_index().rename(columns={'lemma_sr_final': 'lemma', 'pos_sr': 'pos'})

    df_agg[emo_categories] = df_agg[emo_categories].round(2)
    df_agg_emo = df_agg[emo_categories]
    is_max = df_agg_emo.eq(df_agg_emo.max(axis=1), axis=0)

    df_agg[emo_categories] = np.where(df_agg[emo_categories].eq(1), df_agg[df_agg[emo_categories] < 1].max(axis=1).max(axis=0), df_agg[emo_categories])
    df_agg['label'] = is_max.dot(df_agg_emo.columns + " ").str.split(' ').apply(
        lambda x: '|'.join([y.strip() for y in x if y != '']))

    df_agg[emo_categories] = df_agg_emo[is_max].fillna(0)
    df_agg.to_csv(EMOLEX_SR_V1_PATH, sep='\t', index=False)

if __name__ == "__main__":
    main()
