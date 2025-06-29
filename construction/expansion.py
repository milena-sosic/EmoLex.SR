import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from utils import *
from load_lexicons import *
from sklearn.preprocessing import MultiLabelBinarizer
from lexicon_construction.visualization import draw_pie_plot
from lexicon_validation.metrics import measure_accuracy
from constants import *

def correct_synonyms(row):
    def split_and_strip(s):
        """Splits a string by commas and strips whitespace from each element."""
        return [item.strip() for item in s.split(',') if item.strip()]

    synonyms = {group: split_and_strip(row.get(group, '')) for group in SYNONYM_GROUPS}
    
    all_synonyms = synonyms['synonyms_gpt'] + synonyms['synonyms_swn'] + synonyms['synonyms_manual']
    incorrect = set(synonyms['synonyms_incorrect'])
    
    correct_synonyms = list(set(all_synonyms) - incorrect)

    print(row.get('lemma', ''))
    print(correct_synonyms)
    print('|'.join(correct_synonyms))
    print('\n')

    return '|'.join(all_synonyms), '|'.join(correct_synonyms)


def expand_lexicon():
    df = pd.read_csv(EMOLEX_SR_V1_PATH, sep='\t')
    df = df[df['lemma'] != '']

    df_ext = pd.read_csv(CRAFTED_EMO_WORDS_SR, sep=',').rename(columns={'lemma_sr': 'lemma', 'pos_sr': 'pos'})
    df_ext = df_ext[df_ext.valid == 'Yes']
    df_ext[SYNONYM_GROUPS] = ''

    mlb = MultiLabelBinarizer()
    emo_categories = EMO_CATEGORIES + ['neutral']
    
    s1 = df_ext['label'].fillna('').str.split('|').apply(lambda x: [y.strip() for y in x if y in emo_categories])
    mlb_s1 = mlb.fit_transform(s1)
    ann1 = pd.DataFrame(mlb_s1, columns=mlb.classes_, index=df_ext.index)
    df_ext = pd.concat([df_ext, ann1], axis=1)
    df_ext['neutral'] = 0

    df = pd.concat([df, df_ext], axis=0)
    df = df[df['lemma'] != '']
    df[SYNONYM_GROUPS] = df[SYNONYM_GROUPS].fillna('')

    df['synonyms_sr_all'], df['synonyms_sr_correct'] = zip(*df.apply(correct_synonyms, axis=1))

    measure_accuracy(df, type='synonym')

    df_synonyms = df.assign(synonym=df['synonyms_sr_correct'].str.split('|')).explode('synonym')
    df_synonyms = df_synonyms[['synonym', 'pos', 'synonyms_sr_correct', 'label'] + emo_categories].rename(columns={'synonym': 'lemma', 'synonyms_sr_correct': 'synonyms'})

    nlp = load_lemma_pos_model().rename(columns={'lemma_sr': 'lemma', 'pos_sr': 'pos'})
    nlp = nlp.drop_duplicates(subset=['lemma', 'pos'])
    df_synonyms = df_synonyms.merge(nlp, on=['lemma', 'pos'], how='left')
    df_synonyms = df_synonyms[~df_synonyms.word_sr.isna()]

    df = pd.concat([df, df_synonyms], axis=0)
    df = df.drop('synonyms', axis=1)

    _, df['lemma'], _ = zip(
        *df['lemma'].fillna('').apply(lambda x: reflexive_verb(cyrtranslit.to_latin(x).lower())))

    df_agg = df.fillna('').groupby(['lemma', 'pos'])[emo_categories].mean().reset_index()
    df_agg[emo_categories] = df_agg[emo_categories].round(2)

    df_agg_emo = df_agg[emo_categories]
    is_max = df_agg_emo.eq(df_agg_emo.max(axis=1), axis=0)
    df_agg['label'] = is_max.dot(df_agg_emo.columns + " ").str.split(' ').apply(lambda x: '|'.join([y.strip() for y in x if y != '']))
    df_agg[emo_categories] = df_agg_emo[is_max].fillna(0)

    draw_pie_plot(df_agg)

    df_agg.to_csv(EMOLEX_SR_V2_PATH, sep='\t', index=False)

if __name__ == '__main__':
    expand_lexicon()
