import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import os
from constants import *
from prompt_engineering.gpt_prompting import batch_translation
from utils import translate_word_en_to_sr_google, convert_pos_tag
from load_lexicons import validate_lemma_pos
from deep_translator import GoogleTranslator


def calculate_statistics(df, total, valid_col, total_col_name):
    df[valid_col] = df[valid_col].fillna(False)
    correct_count = df[df[valid_col]].shape[0]
    incorrect_count = df[~df[valid_col]].shape[0]
    total_count = df.shape[0]
    correct_percentage = np.round(correct_count / total_count if total_count > 0 else 0, 3)
    incorrect_percentage = np.round(incorrect_count / total_count if total_count > 0 else 0, 3)

    correct_percentage_tot = np.round(correct_count / total if total > 0 else 0, 3)
    incorrect_percentage_tot = np.round(incorrect_count / total if total > 0 else 0, 3)

    print(f"{total_col_name} translation validation:", {total_count})
    print(f"Correct: {correct_count}, {correct_percentage_tot:.3f}%, {correct_percentage:.3f}%")
    print(f"Incorrect: {incorrect_count}, {incorrect_percentage_tot:.3f}%, {incorrect_percentage:.3f}%")


if __name__ == '__main__':

    nrc_en = pd.read_csv(NRC_EN_PATH, sep=',', names=['pos_en', 'word_en', 'label'])
    nrc_en['pos_en'] = nrc_en['pos_en'].apply(convert_pos_tag)
    
    models = ["gpt-3.5-turbo", "gpt-4.1-2025-04-14"]
    gpt_translations = nrc_en[['word_en', 'pos_en']]
    for model in models:
        TRANSLATION_GPT_PATH = f"./data/{model}/NRC.EN.tr.gpt.tsv"
        suffix = '3_5' if model == "gpt-3.5-turbo" else '4_1'
        if os.path.exists(TRANSLATION_GPT_PATH):
            nrc_en_gpt = pd.read_csv(TRANSLATION_GPT_PATH, sep='\t')
        else:
            nrc_en_gpt = batch_translation(NRC_EN_PATH, TRANSLATION_GPT_PATH)
            nrc_en_gpt = validate_lemma_pos(nrc_en_gpt, type='gpt')
            nrc_en_gpt.to_csv(TRANSLATION_GPT_PATH, sep='\t', index=False)      
        gpt_translations = gpt_translations.merge(nrc_en_gpt[['word_en', 'pos_en', 'word_sr', 'pos_sr', 'gpt_PoS_valid', 'gpt_lemma_valid', 'gpt_valid']]
                                                  .rename(columns={'word_sr': f'word_sr_{suffix}', 'pos_sr': f'pos_sr_{suffix}', 
                                                  'gpt_PoS_valid': f'gpt_PoS_valid_{suffix}', 'gpt_lemma_valid': f'gpt_lemma_valid_{suffix}', 'gpt_valid': f'gpt_valid_{suffix}'}), 
                                                  on=['word_en', 'pos_en'], how='left')

    if os.path.exists(TRANSLATION_GT_PATH):
        nrc_en_gt = pd.read_csv(TRANSLATION_GT_PATH, sep='\t')
    else:
        translator = GoogleTranslator(source='auto', target='sr')
        nrc_en['word_gt_sr'] = nrc_en['word_en'].apply(lambda x: translate_word_en_to_sr_google(translator, str(x)))
        nrc_en_gt = nrc_en
        nrc_en_gt = validate_lemma_pos(nrc_en_gt, type='gt')
        nrc_en_gt.to_csv(TRANSLATION_GT_PATH, sep='\t', index=False)
        
    nrc_en_gt = nrc_en_gt[['word_en', 'pos_en', 'word_sr', 'pos_sr', 'gt_PoS_valid', 'gt_lemma_valid', 'gt_valid']].rename(columns={'word_sr': f'word_sr_gt', 'pos_sr': f'pos_sr_gt'})

    nrc_en_tr = nrc_en_gt.merge(gpt_translations, how='left', on=['word_en', 'pos_en'])
    nrc_en_tr = nrc_en.merge(nrc_en_tr, how='left', on=['word_en', 'pos_en']) 
    nrc_en_tr[['gt_valid', 'gpt_valid_3_5', 'gpt_valid_4_1']] = nrc_en_tr[['gt_valid', 'gpt_valid_3_5', 'gpt_valid_4_1']].fillna(False)

    print(nrc_en_tr[nrc_en_tr.gt_valid | nrc_en_tr.gpt_valid_3_5 | nrc_en_tr.gpt_valid_4_1].shape[0]/nrc_en_tr.shape[0])
    # Calculate statistics
    calculate_statistics(nrc_en_tr, nrc_en_tr.shape[0], 'gt_valid', "GT")
    calculate_statistics(nrc_en_tr, nrc_en_tr.shape[0], 'gt_PoS_valid', "GT-PoS")
    calculate_statistics(nrc_en_tr, nrc_en_tr.shape[0], 'gt_lemma_valid', "GT-Lemma")
    calculate_statistics(nrc_en_tr[~nrc_en_tr.gpt_valid_3_5], nrc_en_tr.shape[0], 'gt_valid', "GT/ChatGPT Not Valid")
    calculate_statistics(nrc_en_tr[nrc_en_tr.gpt_valid_3_5], nrc_en_tr.shape[0], 'gt_valid', "GT/ChatGPT Valid")
    calculate_statistics(nrc_en_tr[~nrc_en_tr.gpt_valid_4_1], nrc_en_tr.shape[0], 'gt_valid', "GT/GPT Not Valid")
    calculate_statistics(nrc_en_tr[nrc_en_tr.gpt_valid_4_1], nrc_en_tr.shape[0], 'gt_valid', "GT/GPT Valid")
    calculate_statistics(nrc_en_tr, nrc_en_tr.shape[0], 'gpt_valid_3_5', "ChatGPT")
    calculate_statistics(nrc_en_tr, nrc_en_tr.shape[0], 'gpt_PoS_valid_3_5', "ChatGPT-PoS")
    calculate_statistics(nrc_en_tr, nrc_en_tr.shape[0], 'gpt_lemma_valid_3_5', "ChatGPT-Lemma")
    calculate_statistics(nrc_en_tr[~nrc_en_tr.gt_valid], nrc_en_tr.shape[0], 'gpt_valid_3_5', "ChatGPT/GT Not Valid")
    calculate_statistics(nrc_en_tr[nrc_en_tr.gt_valid], nrc_en_tr.shape[0], 'gpt_valid_3_5', "ChatGPT/GT Valid")
    calculate_statistics(nrc_en_tr[~nrc_en_tr.gpt_valid_4_1], nrc_en_tr.shape[0], 'gpt_valid_3_5', "ChatGT/GPT Not Valid")
    calculate_statistics(nrc_en_tr[nrc_en_tr.gpt_valid_4_1], nrc_en_tr.shape[0], 'gpt_valid_3_5', "ChatGT/GPT Valid")
    calculate_statistics(nrc_en_tr, nrc_en_tr.shape[0], 'gpt_valid_4_1', "GPT")
    calculate_statistics(nrc_en_tr, nrc_en_tr.shape[0], 'gpt_PoS_valid_4_1', "GPT-PoS")
    calculate_statistics(nrc_en_tr, nrc_en_tr.shape[0], 'gpt_lemma_valid_4_1', "GPT-Lemma")
    calculate_statistics(nrc_en_tr[~nrc_en_tr.gt_valid], nrc_en_tr.shape[0], 'gpt_valid_4_1', "GPT/GT Not Valid")
    calculate_statistics(nrc_en_tr[nrc_en_tr.gt_valid], nrc_en_tr.shape[0], 'gpt_valid_4_1', "GPT/GT Valid")
    calculate_statistics(nrc_en_tr[~nrc_en_tr.gpt_valid_3_5], nrc_en_tr.shape[0], 'gpt_valid_4_1', "GPT/ChatGPT Not Valid")
    calculate_statistics(nrc_en_tr[nrc_en_tr.gpt_valid_3_5], nrc_en_tr.shape[0], 'gpt_valid_4_1', "GPT/ChatGPT Valid")


    df = pd.read_csv(EMOLEX_SR_VAL_PATH, sep='\t')
    nrc_en_tr = nrc_en_tr.merge(df[['word_en', 'pos_en', 'word_sr_fin'] + SYNONYM_GROUPS], on=['word_en', 'pos_en'], how='left')

    print(nrc_en_tr[(nrc_en_tr.word_sr_3_5 == nrc_en_tr.word_sr_fin)].shape[0]/nrc_en_tr.shape[0])
    print(nrc_en_tr[(nrc_en_tr.word_sr_4_1 == nrc_en_tr.word_sr_fin)].shape[0]/nrc_en_tr.shape[0])

    nrc_en_tr.to_csv(NRC_EN_TR_PATH, sep='\t', index=False)
    

