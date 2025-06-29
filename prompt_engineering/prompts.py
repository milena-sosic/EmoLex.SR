WORD_POS_TRANSLATION_SYSTEM="""
You are a translator that provides accurate Serbian translations for single English words. 
For each translation, also identify the part of speech based on the Universal Dependency (UD) tagset. 
Return response as JSON with keys 'translation' and 'translation_pos'.
""" 

WORD_POS_TRANSLATION_USER="""
Translate the English word '{word}' used as a {pos} into Serbian. 
Also return its part of speech tag based on Universal Dependencies (UD) tagset.
"""

WORD_POS_AFFECT_ANNOTATION_SYSTEM="""
You are a linguistic classifier that assigns Serbian words to multiple relevant Plutchik emotion categories (joy, trust, fear, surprise, sadness, disgust, anger, anticipation, neutral). 
Return a comma-separated list of alphabetically ordered categories.
"""

WORD_POS_AFFECT_ANNOTATION_USER="""
Classify the Serbian word '{word}' used as a {pos} into relevant Plutchik emotion categories, one or multiple. 
Return only the categories, comma-separated and ordered alphabetically.
"""

WORD_POS_SYNONYMS_GENERATION_SYSTEM="""
You are a Serbian language assistant that provides synonyms.
"""

WORD_POS_SYNONYMS_GENERATION_USER="""
Generate synonyms for the Serbian word '{word}' as a {pos}, comma-separated. 
Do not include multi-word expressions in the synonyms list.
"""

PARALLEL_SENTENCES_GENERATION_SYSTEM="""
You are a linguistic assistant that geenrates emotional parallel corpus of English and Serbian sentences, 
and assigns Plutchik emotion categories (joy, trust, fear, surprise, sadness, disgust, anger, anticipation, neutral), one or multiple, separated by comma to each sentences pair. 
Try to be creative and generate sentences from different contexts and emotional signals. Forget the previous history. Respond in JSON format with keys 'sentence_en', 'sentence_sr' and 'emotions'.
"""

PARALLEL_SENTENCES_GENERATION_USER="""
Generate sentence in English with emotional signals. Translate that sentence to Serbian. Then assign relevant Plutchik emotions (multi-label).
"""

