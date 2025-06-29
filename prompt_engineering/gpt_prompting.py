# TASKS:
# 1. English (word, PoS) translation to Serbian (EN->SR)
# 2. Serbian (lemma, PoS) emotion annotation (multi-label)
# 3. Serbian (lemma, PoS) synonyms generation
# 4. Parallel corpus of EN/SR sentences annotated into Plutchik's categories (multi-label)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from typing import List, Tuple, Dict
from openai import OpenAI
from collections import Counter
import re
from prompt_engineering.prompts import *
from constants import *
from utils import convert_pos_tag, affect_label

import logging

logger = logging.getLogger(__name__)

# openai.api_key = os.getenv("OPENAI_API_KEY") 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# 1. English (word, PoS) translation (EN -> SR) for the given number of iterations + majority voting
def translate_word_en_to_sr_gpt(word: str, pos: str, iterations:int = 1) -> Dict[str, str]:
    try:
        system_message = WORD_POS_TRANSLATION_SYSTEM
        user_message = WORD_POS_TRANSLATION_USER.format(word=word, pos=pos)
        translations = []
        for _ in range(iterations):
            response = client.chat.completions.create(
                model=GPT_MODEL,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            output = response.choices[0].message.content.strip()

            # Parse to JSON
            try:
                json_output = eval(output) if output.startswith('{') else None
                if isinstance(json_output, dict) and 'translation' in json_output and 'translation_pos' in json_output:
                    translations.append((json_output['translation'].strip(), json_output['translation_pos'].strip()))
                    continue
            except:
                logger.error(f"Failed to covert LLM output to JSON, error: {str(e)}")
                raise

            # Fallback regex parsing
            match = re.match(r"(.+?)\s*///\s*(\w+)", output)
            if match:
                translation, sr_pos = match.groups()
                translations.append((translation.strip(), sr_pos.strip().lower()))

        if not translations:
            return {"translation": "", "translation_pos": "", "translations": []}

        # Majority voting by translation only
        translation_counts = Counter([t[0] for t in translations])
        best_translation = translation_counts.most_common(1)[0][0]
        matching_pos = [t[1] for t in translations if t[0] == best_translation]
        best_pos = Counter(matching_pos).most_common(1)[0][0] if matching_pos else ""
        return {"translation": best_translation, "translation_pos": best_pos, "translations": translations}
    except Exception as e:
            logger.error(f"Failed to translate {word}, {pos}, error: {str(e)}")
            raise

def batch_translation(filepath: str, output_path: str):
    df = pd.read_csv(filepath)
    # Open the output file in append mode
    with open(output_path, 'a', newline='', encoding='utf-8') as file:
        # Write the header if the file is empty
        if file.tell() == 0:
            file.write("word_en\tpos_en\tword_sr\tpos_sr\tword_sr_1\tpos_sr_1\tword_sr_2\tpos_sr_2\tword_sr_3\tpos_sr_3\n")

        for _, row in df.iterrows():
            word = str(row['word'])
            pos = convert_pos_tag(str(row['pos']))
            try:
                gpt_result = translate_word_en_to_sr_gpt(word, pos, iterations=3)
                sr_translation_gpt = gpt_result['translation']
                sr_pos = gpt_result['translation_pos']
                translations = gpt_result['translations']

                # Prepare the row for writing
                row_data = [
                    word, pos, sr_translation_gpt, sr_pos.upper(),
                    *(item for sublist in translations[:3] for item in sublist)
                ]
                # Ensure the row has exactly 10 elements
                row_data += [''] * (10 - len(row_data))

                # Write the row to the file
                file.write('\t'.join(row_data) + '\n')
                file.flush()
            except Exception as e:
                print(f"Error raised for {word}, {pos}, error: {e}")

    print(f"Translations saved at: {output_path}")


# 2. Serbian (lemma, PoS) emotion annotation (multi-label)
def classify_emotion_llm(word: str, pos: str, iterations:int = 1) -> str:
    try:
        system_message = WORD_POS_AFFECT_ANNOTATION_SYSTEM
        user_message = WORD_POS_AFFECT_ANNOTATION_USER.format(word=word, pos=pos)
        labels = pd.DataFrame()
        for i in range(iterations):
            response = client.chat.completions.create(
                model=GPT_MODEL,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )
            labels[f'label_{i}'] = response.choices[0].message.content.strip()

        gpt_label = affect_label(labels, ['label_1', 'label_2', 'label_3'])

        return {'gpt_label': gpt_label, 'labels': labels.to_list()}
    except Exception as e:
            logger.error(f"Failed to annotate {word}, {pos}, error: {str(e)}")
            raise


def batch_affect_annotations(filepath: str, output_path: str):
    df = pd.read_csv(filepath)
    with open(output_path, 'a', newline='', encoding='utf-8') as file:
        # Write the header if the file is empty
        if file.tell() == 0:
            file.write("word_sr\tpos_sr\tgpt_label\tlabel_1\tlabel_2\tlabel_3\n")

        for _, row in df.iterrows():
            word = str(row['word_sr'])
            pos = convert_pos_tag(str(row['pos_sr']))
            try:
                gpt_result = classify_emotion_llm(word, pos, iterations=3)
                emotion = gpt_result['gpt_label']
                labels = gpt_result['labels']
                
                # Prepare the row for writing
                row_data = [
                    word, pos, emotion, 
                    *(item for item in labels[:3])
                ]
                # Ensure the row has exactly 10 elements
                row_data += [''] * (6 - len(row_data))

                # Write the row to the file
                file.write('\t'.join(row_data) + '\n')
                file.flush()
            except Exception as e:
                print(f"Error raised for {word}, {pos}, error: {e}")
            
    print(f"Affect annotations saved at: {output_path}")


# 3. Serbian (lemma, PoS) synonyms generation
def generate_synonyms(word: str, pos: str) -> List[str]:
    try:
        system_message = WORD_POS_SYNONYMS_GENERATION_SYSTEM
        user_message = WORD_POS_SYNONYMS_GENERATION_USER.format(word=word, pos=pos)
        response = client.chat.completions.create(
            model=GPT_MODEL,
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        synonyms = response.choices[0].message.content.strip().split(',')
        return [s.strip() for s in synonyms]
    except Exception as e:
            logger.error(f"Failed to generate synonyms for {word}, {pos}, error: {str(e)}")
            raise
    

def batch_synonyms_generation(filepath: str, output_path: str):
    df = pd.read_csv(filepath)
    with open(output_path, 'a', newline='', encoding='utf-8') as file:
        # Write the header if the file is empty
        if file.tell() == 0:
            file.write("word_sr\tpos_sr\tgpt_synonyms\n")

        for _, row in df.iterrows():
            word = str(row['word_sr'])
            pos = convert_pos_tag(str(row['pos_sr']))
            try:
                gpt_synonyms = generate_synonyms(word, pos)
                # Prepare the row for writing
                row_data = [
                    word, pos, gpt_synonyms
                ]
                # Ensure the row has exactly 10 elements
                row_data += [''] * (3 - len(row_data))

                # Write the row to the file
                file.write('\t'.join(row_data) + '\n')
                file.flush()
            except Exception as e:
                print(f"Error raised for {word}, {pos}, error: {e}")

    print(f"Generated synonyms saved at: {output_path}")


# 4. Parallel corpus of EN/SR sentences annotated into Plutchik's categories (multi-label)
def create_parallel_corpus(output_path: str):
    try:
        with open(output_path, 'a', newline='', encoding='utf-8') as file:
            # Write the header if the file is empty
            if file.tell() == 0:
                file.write("sentence_en\tsentence_sr\temotions\n")
            results = []
            system_message = PARALLEL_SENTENCES_GENERATION_SYSTEM
            for i in range(1000):
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                user_message = PARALLEL_SENTENCES_GENERATION_SYSTEM
                response = client.chat.completions.create(
                model=GPT_MODEL,
                temperature=1,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ]
                )
                content = response.choices[0].message.content.strip()
                
                data = eval(content) if content.startswith('{') else None
                if isinstance(data, dict) and 'sentence_sr' in data and 'emotions' in data:
                    # Prepare the row for writing
                    row_data = [
                        data['sentence_en'], data['sentence_sr'], data['emotions']
                    ]
                    # Ensure the row has exactly 10 elements
                    row_data += [''] * (3 - len(row_data))

                    # Write the row to the file
                    file.write('\t'.join(row_data) + '\n')
                    file.flush()
    except Exception as e:
            logger.error(f"Failed to generate parralel corpus, error: {str(e)}")
            raise


if __name__ == "__main__":
    
    batch_translation(NRC_EN_PATH, TRANSLATION_GPT_PATH)
    batch_affect_annotations(TRANSLATION_GPT_PATH, AFFECT_LABELS_GPT_PATH)
    batch_synonyms_generation(TRANSLATION_GPT_PATH, SYNONYMS_GPT_PATH)
    create_parallel_corpus(PARALLEL_GPT_PATH)
