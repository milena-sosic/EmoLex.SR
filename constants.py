EMO_CATEGORIES = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness','surprise', 'trust']

SYNONYM_GROUPS = ['synonyms_gpt', 'synonyms_swn', 'synonyms_incorrect', 'synonyms_manual']

NRC_WNA_MAPPING = {
        'fear': ['ambiguous-fear', 'gravity', 'daze', 'shame', 'anxiety', 'negative-fear', 'scare', 'horror', 'shyness', 'timidity', 'diffidence', 'annoyance', 'negative-concern'],
        'sadness': ['apathy', 'pensiveness', 'compassion', 'sadness', 'melancholy', 'regret-sorrow', 'grief', 'lost-sorrow'],
        'anger': ['despair', 'ingratitude', 'general-dislike', 'anger', 'annoyance', 'bad-temper', 'oppression', 'hate', 'displeasure', 'bad-temper'],
        'anticipation': ['ambiguous-expectation', 'positive-expectation', 'positive-hope', 'forgiveness'],
        'surprise': ['ambiguous-agitation', 'surprise', 'stupefaction', 'astonishment'],
        'disgust': ['neutral-unconcern', 'thing', 'ingratitude', 'general-dislike', 'disgust', 'antipathy', 'dislike', 'repugnance'],
        'trust': ['fearlessness', 'self-pride', 'humility', 'encouragement', 'approval', 'confidence', 'security', 'belonging', 'closeness', 'favor'],
        'joy': ['levity', 'positive-fear', 'enthusiasm', 'calmness', 'joy', 'gratitude', 'affection', 'love', 'liking', 'happiness', 'joy-pride', 'cheerfulness', 'euphoria', 'satisfaction', 'general-gaiety', 'jollity', 'sympathy', 'positive-concern']
    }

CORPUS = ['SocialSCKT'] #'XED', 'LLM',  'LLM'

XLM_MODEL_PATH = "MilaNLProc/xlm-emo-t"
EMO_LLAMA_MODEL_PATH = "lzw1008/Emollama-chat-7b"
SBERT_MODEL_PATH = "sentence-transformers/use-cmlm-multilingual"

GPT_MODEL = "gpt-4.1-2025-04-14" #"gpt-3.5-turbo"

# Negation signals, negation modifiers, and adverb modifiers for Serbian
NEGATION_SIGNALS_SR = ['ne', 'ni']
NEGATION_MODIFIERS_SR = ['nikada', 'nema', 'nije']
ADVERB_MODIFIERS_SR = ['vrlo', 'veoma', 'izuzetno', 'ekstremno', 'totalno', 'neverovatno', 'sasvim', 'stvarno', 'pošteno', 'iskreno', 'pravično', 'pravedno', 'zakonito', 'zaista', 'ozbiljno', 'strašno', 'užasno']
HELPER_VERBS_SR = ['biti', 'hteti']
MORHO_NEG_PREFIXES = ['ne', 'bez', 'ni', 'a', 'dis', 'in']

# Negation signals, negation modifiers, and adverb modifiers for English
NEGATION_SIGNALS_EN = ['not', 'no']
NEGATION_MODIFIERS_EN = ['never', 'none']
ADVERB_MODIFIERS_EN = ['very', 'extremely', 'highly', 'greatly', 'absolutely', 'totally', 'fairly', 'tremendously', 'incredibly', 'quite', 'rather', 'really', 'truly', 'seriously', 'completely', 'utterly', 'thoroughly', 'awfully', 'horribly', 'rightful']
HELPER_VERBS_EN = ['be', 'have', 'do']

# Punctuation marks used to identify sentence boundaries
SENTENCE_BOUNDARIES = set(['.', ',', ';', ':', '!', '?'])

NRC_EMOINT_PATH = './lexicons/NRC.EmoInt.tr'
NRC_EN_PATH = './lexicons/NRC.EN.csv'
NRC_EN_TR_PATH = './lexicons/NRC.EN.tr.csv'
EMOLEX_SR_VAL_PATH = './lexicons/EmoLex.SR.val.csv'
EMOLEX_SR_V1_PATH = './lexicons/EmoLex.SR-v1.csv'
EMOLEX_SR_V2_PATH = './lexicons/EmoLex.SR-v2.csv'

LPT_PATH = "./lexicons/lexiconPOSlat"
SWN_PATH = "./WNA.SR/data/wnsrp.xml"
SWN_LEMMA_PATH = "./WNA.SR/data/wnsrp-lemma.csv"
WNA_LEMMA_PATH = "./WNA.SR/data/wna-lemma.csv"
SWNA_PATH = "./WNA.SR/data/wnsrp-wna.xml"
WN16_PATH = "./WNA.SR/wordnet-1.6/dict"
WN30_PATH = "./WNA.SR/WordNet-3.0/dict"
AFF_SYNSETS_PATH_EN = "./WNA.SR/wn-domains-3.2/wn-affect-1.1/a-synsets.xml"
AFF_SYNSETS_PATH_SR = "./WNA.SR/wn-domains-3.2/wn-affect-1.1/a-synsets-sr.xml"
RELDIA_SR_LEX = "./lexicons/wikitweetweb.sr.tm"

TRANSLATION_GPT_PATH = f"./data/{GPT_MODEL}/NRC.EN.tr.gpt.tsv"
AFFECT_LABELS_GPT_PATH = f"./data/{GPT_MODEL}/NRC.EN.label.gpt.tsv"
SYNONYMS_GPT_PATH = f"./data/{GPT_MODEL}/NRC.EN.synonyms.gpt.tsv"
PARALLEL_GPT_PATH = f"./data/{GPT_MODEL}/llm_emo.gpt.tsv"
TRANSLATION_GT_PATH = f"./data/gt/NRC.EN.tr.gt.tsv"


STOP_WORDS_PATH = './data/stop_words_sr.csv'
CRAFTED_EMO_WORDS_SR = './data/serbian_emo_affect_words.csv'

