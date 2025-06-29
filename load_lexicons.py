import pandas as pd
import numpy as np
import cyrtranslit
from constants import *
from utils import *
from constants import EMO_CATEGORIES
import logging

logger = logging.getLogger(__name__)

def load_stop_words():
    stop_df = pd.read_csv(STOP_WORDS_PATH, sep=',')
    stop_words = list(stop_df['word_lat'].values)
    return stop_words


def load_sr_wn_affect():
    try:
        srpwpn = pd.read_csv(WNA_LEMMA_PATH, sep='\t')

        # Ensure 'srpwords' is a string before applying eval
        srpwpn['srpwords'] = srpwpn['srpwords'].apply(lambda x: eval(x) if isinstance(x, str) else x)

        # Explode the 'srpwords' column which contains a list of dictionaries
        srpwpn_exploded = srpwpn.explode('srpwords').reset_index(drop=True)

        # Normalize the 'srpwords' dictionaries into separate columns
        srpwords_normalized = pd.json_normalize(srpwpn_exploded['srpwords']).reset_index(drop=True)

        # Concatenate the normalized columns back to the original DataFrame
        srpwpn_expanded = pd.concat([srpwpn_exploded.drop(columns=['srpwords']), srpwords_normalized], axis=1)

        # Apply the mapping to convert WordNet tags
        srpwpn_expanded['srp-pos'] = srpwpn_expanded['srp-pos'].fillna('').apply(convert_wordnet_tag)
        
        # Group by the necessary columns and aggregate
        srpwpn_grp = srpwpn_expanded.groupby(['srp-word', 'srp-pos'])[['categ', 'srp-gloss']].agg(','.join).reset_index() \
            .rename(columns={'srp-pos': 'pos_sr', 'srp-word': 'lemma_sr', 'srp-gloss': 'gloss_sr', 'categ': 'wna_label'})

        return srpwpn_grp
    except Exception as e:
            logger.error(f"Failed to load WNA.SR lexicon, error: {str(e)}")
            raise


def load_sr_wn():
    try:
        wnsrp = pd.read_csv(SWN_LEMMA_PATH, sep='\t')
        wnsrp = wnsrp[['lemma', 'pos', 'sim_words']].rename(
            columns={'lemma': 'word_sr', 'pos': 'pos_sr', 'sim_words': 'synonyms'})

        wnsrp['pos_sr'] = wnsrp['pos_sr'].apply(convert_wordnet_tag)
        wnsrp = wnsrp.fillna('')
        wnsrp = wnsrp[wnsrp.synonyms != '']
        wnsrp = wnsrp.groupby(['word_sr', 'pos_sr'])[['synonyms']].agg(','.join).reset_index(drop=False)
        wnsrp['synonyms'] = wnsrp['synonyms'].apply(lambda x: x.strip(','))
        return wnsrp
    except Exception as e:
            logger.error(f"Failed to load SWN lexicon, error: {str(e)}")
            raise


def load_lemma_pos_model(level=1):
    try:
        nlp = pd.read_csv(LPT_PATH, sep='\t', names=['word', 'pos_lemma', 'ext', '1', '2', '3'], low_memory=False)
        nlp[['pos', 'lemma']] = nlp['pos_lemma'].str.split(' ', n=1, expand=True).rename(columns={0: 'pos', 1: 'lemma'})
        nlp[['pos1', 'lemma1']] = nlp['ext'].str.split(' ', n=1, expand=True).rename(columns={0: 'pos', 1: 'lemma'})
        nlp[['pos2', 'lemma2']] = nlp['1'].str.split(' ', n=1, expand=True).rename(columns={0: 'pos', 1: 'lemma'})
        mapping = {
            'N': 'NOUN',
            'A': 'ADJ',
            'ADV': 'ADV',
            'V': 'VERB',
            'INT': 'INT',
            'X': 'X'
        }
        nlp1 = nlp[['word', 'pos', 'lemma']].rename(columns={'pos': 'pos', 'lemma': 'lemma'})
        if level > 1:
            nlp2 = nlp[['word', 'pos1', 'lemma1']].rename(columns={'pos1': 'pos', 'lemma1': 'lemma'})
            nlp1 = pd.concat([nlp1, nlp2], axis=0)
        if level == 3:
            nlp3 = nlp[['word', 'pos2', 'lemma2']].rename(columns={'pos2': 'pos', 'lemma2': 'lemma'})
            nlp1 = pd.concat([nlp1, nlp3], axis=0)
        
        nlp = nlp1[['word', 'pos', 'lemma']]
        nlp['pos'] = nlp['pos'].map(mapping)
        nlp = nlp[~nlp['pos'].isna()]
        #nlp = nlp.rename(columns={'word': 'word_sr', 'pos': 'pos_sr', 'lemma': 'lemma_sr'})
        return nlp
    except Exception as e:
            logger.error(f"Failed to load Lemma/PoS Tagger lexicon, error: {str(e)}")
            raise


def load_sr_lexicon(version=1):
    try:
        lexiconPath = EMOLEX_SR_V1_PATH if version == 1 else EMOLEX_SR_V2_PATH
        df = pd.read_csv(lexiconPath, sep='\t', encoding="utf8")
        # df = df[['lemma', 'pos'] + EMO_CATEGORIES + ['neutral']].drop_duplicates()
        # df.dropna(inplace=True)
        return df
    except Exception as e:
            logger.error(f"Failed to load EmoLex.SR-v{version} lexicon, error: {str(e)}")
            raise


def load_emoint(lang='sr'):
    try:
        column = 'Serbian Word'
        if lang == 'en':
            column = 'English Word'
        df = pd.read_csv(NRC_EMOINT_PATH, sep='\t', encoding="utf8").rename(columns={column: 'lemma'})
        if lang == 'sr':
             df['lemma'] = df['lemma'].apply(cyrtranslit.to_latin)
        df_emo = df[EMO_CATEGORIES]
        # is_max = df.eq(1, axis=0)
        is_max = df_emo.gt(0)
        result = is_max.dot(df_emo.columns + " ").str.split(' ').apply(lambda x: ','.join([y.strip(' ') for y in x if y != '']))
        ann = pd.concat([df, pd.DataFrame(result, columns=['label'])], axis=1)
        ann = ann[ann.label != '']
        ann = ann[['lemma'] + EMO_CATEGORIES].drop_duplicates()
        ann.dropna(inplace=True)
        return ann
    except Exception as e:
            logger.error(f"Failed to load EmoLex translated lexicon, error: {str(e)}")
            raise


def validate_lemma_pos(df, type='gt'):
    try:
        df['word_sr_full'], df[f'word_sr'], df['reflexive'] = zip(
            *df[f'word_sr'].fillna('').apply(lambda x: reflexive_verb(cyrtranslit.to_latin(x).lower())))

        nlp = load_lemma_pos_model(level=3).rename(columns={'word': 'word_sr', 'pos': 'pos_sr', 'lemma': 'lemma_sr'})
        df['pos_sr'] = df['pos_sr'] if type in ['gpt', 'gpt4'] else df['pos_en']

        df = df.merge(nlp, on=[f'word_sr', f'pos_sr'], how='left')
        df['pos_sr'] = np.where(df['word_sr'].str.split(' ').str.len() > 1, 'MWE', df['pos_sr'])
        
        equal_PoS = np.where((df['pos_en'] == df['pos_sr']) | (df['pos_sr'] == 'MWE'), True, False)

        non_adj_form = (df['pos_sr'] != 'ADJ') & (~df['lemma_sr'].isna()) & (df['lemma_sr'] == df['word_sr'])
        valid_lemma = (~df['lemma_sr'].isna()) & (df['lemma_sr'] == df['word_sr'])
        adj_form = (df['pos_sr'] == 'ADJ') & (df['lemma_sr'] == df['word_sr'])
        correct_lemma = np.where(adj_form | non_adj_form, True, False)
        # adj_adv_form = np.where((df['pos_sr'] == 'ADJ') & (df['lemma_sr'] == df['word_sr']), True, False)
        #correct_lemma = np.where(~df['lemma_sr'].isna(), True, False)
        df[f'{type}_PoS_valid'] = equal_PoS
        df[f'{type}_lemma_valid'] = valid_lemma
        df[f'{type}_valid'] = equal_PoS & correct_lemma
        return df
    except Exception as e:
            logger.error(f"Failed to validate lemma/PoS, error: {str(e)}")
            raise


def validate_pos(df, type='gt'):
    if type == 'gt':
        pos_column = 'pos_sr'
    else:
        pos_column = 'pos_sr_winner'
    df[f'{type}_valid'] = np.where(df['pos_en'] == df[pos_column], True, False)

    return df

if __name__ == "__main__":
    load_sr_wn_affect()
    print('completed')