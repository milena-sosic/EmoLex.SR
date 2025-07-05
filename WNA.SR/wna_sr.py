import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import *
from xml.dom import minidom
from functools import reduce
import pandas as pd
from nltk.corpus import WordNetCorpusReader
import nltk
nltk.download('dict')

from constants import NRC_WNA_MAPPING, SWN_PATH, AFF_SYNSETS_PATH_EN, AFF_SYNSETS_PATH_SR, SWNA_PATH, WN16_PATH, WN30_PATH

cwd = os.getcwd()
os.environ["NLTK_DATA"] = cwd
WN16 = WordNetCorpusReader(os.path.abspath("{0}/{1}".format(cwd, WN16_PATH)), None)
WN = WordNetCorpusReader(os.path.abspath("{0}/{1}".format(cwd, WN30_PATH)), None)

# Load WordNet-Affect synsets
def load_asynsets(corpus):
    tree = ET.parse(corpus)
    root = tree.getroot()

    asynsets = {}
    for pos in ["noun", "adj", "verb", "adv"]:
        asynsets[pos] = {}
        for elem in root.findall(".//%s-syn-list//%s-syn" % (pos, pos)):
            (p, offset) = elem.get("id").split("#")
            if not offset: continue

            asynsets[pos][offset] = { "offset16": offset, "pos": pos };
            if elem.get("categ"):
                asynsets[pos][offset]["categ"] = elem.get("categ")
            if elem.get("noun-id"):
                noun_offset = elem.get("noun-id").replace("n#", "", 1)
                asynsets[pos][offset]["noun-offset"] = noun_offset
                asynsets[pos][offset]["categ"] = asynsets["noun"][noun_offset]["categ"]
            if elem.get("caus-stat"):
                asynsets[pos][offset]["caus-stat"] = elem.get("caus-stat")

    return asynsets


def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def output_srp_wnet(s, corpus):
    tree = ET.parse(corpus)
    root = tree.getroot()
    for synset in root.findall('SYNSET'):
        row = {}
        pos = synset.find('POS').text
        id = synset.find('ID').text.replace("ENG30-", "")
        df = s[s['db-synset'] == id]
        if not df.empty:
            affect = ET.Element('AFFECT')
            affect.text = ','.join(list(set(df['categ'].tolist())))
            affect.tail = "\n   "
            synset.append(affect)

            emocat = ET.Element('EMOCAT')
            emocat.text = ','.join(list(set(df['emo_categ'].tolist())))
            emocat.tail = "\n   "
            synset.append(emocat)
    indent(root)
    tree.write(SWNA_PATH, encoding="utf-8", xml_declaration=True)


def load_sr_synsets(corpus, type=''):
    tree = ET.parse(corpus)
    root = tree.getroot()

    asynsets = {}
    pos_map = {"n": "n", "a": "a", "v": "v", "b": "r"}
    rows = []
    for synset in root.findall('SYNSET'):
        row = {}
        pos = synset.find('POS').text
        id = synset.find('ID').text.replace("ENG30-", "")
        positive = synset.find('SENTIMENT/POSITIVE').text
        negative = synset.find('SENTIMENT/NEGATIVE').text
        domain = ''
        if synset.find('DOMAIN') != None:
            domain = synset.find('DOMAIN').text
        affect = ''
        if synset.find('AFFECT') != None:
            a = synset.find('AFFECT').text.split(',')
            a = [i.replace(" ", "") for i in a if i != ''] 
            b = [k for k, v in NRC_WNA_MAPPING.items() for i in a if i in v]
            affect = '|'.join(list(set(b)))
        gloss = ''
        if synset.find('DEF') != None:
            gloss = synset.find('DEF').text
        print(pos, id)
        synonyms = synset.findall('SYNONYM/LITERAL')
        syn_list = [l.text for l in synonyms]
        for literal in synonyms:
            print(literal.text)
            words_list = [w for w in syn_list if w != literal.text]
            print(words_list)
            row = {
                'word': literal.text,
                'pos': pos,
                'db-synset': id.strip(),
                'gloss': gloss,
                'sim_words': ','.join(words_list),
                'domain': domain,
                'affect': affect,
                'positive': positive,
                'negative': negative
            }
            rows.append(row)

    pd.DataFrame(rows).to_csv(SWN_PATH.replace('xml', 'csv'), sep='\t', index=False)
    return

# Merge WordNet-Affect synsets with WordNet-3.0 synsets
def merge_asynset_with_wn(asynsets):
    pos_map = { "noun": "n", "adj": "a", "verb": "v", "adv": "r" }
    for pos in ["noun", "adj", "verb", "adv"]:
        for offset in asynsets[pos].keys():
            synset_16 = WN16.synset_from_pos_and_offset(pos_map[pos], int(offset))
            if not synset_16: continue

            synset_30 = _wn30_synsets_from_wn16_synset(synset_16)
            if not synset_30:
                asynsets[pos][offset]["missing"] = 1
            else:
                (word, p, index) = synset_30.name().split(".")
                asynsets[pos][offset]["word"] = word
                asynsets[pos][offset]["synset"] = synset_30.name()
                asynsets[pos][offset]["db-synset"] = str("%08d-%s" % (synset_30.offset(), p))
                asynsets[pos][offset]["offset30"] = str("%08d" % (synset_30.offset()))
                if "noun-offset" in asynsets[pos][offset]:
                    noffset = asynsets[pos][offset]["noun-offset"]
                    asynsets[pos][offset]["noun-synset"] = asynsets["noun"][noffset]["synset"]

    return asynsets

# Get WordNet-3.0 synset
# Similarity is calculated by wup_similarity
def _wn30_synsets_from_wn16_synset(synset):
    (word, p, index) = synset.name().split(".")
    if p == 's':
        p = 'a'
    synsets = WN.synsets(word, p)
    # print(synsets)
    if len(synsets) == 0: return

    synset_sims = {}
    for i in range(len(synsets)):
        try:
            synset_sims[i] = synset.wup_similarity(synsets[i])
            # fallback to 0 when similarity is None
            if synset_sims[i] == None:
                synset_sims[i] = 0
        except (RuntimeError, TypeError, NameError):
            # Set similarity to 0 in case of RuntimeError
            synset_sims[i] = 0
    # Most similar synset index
    index = sorted(synset_sims.items(), key=lambda x:x[1], reverse=True)[0][0]
    # print(synsets[index])
    return synsets[index]

# Merge asynsets with Serbian WordNet
def merge_asynset_with_wnsrp(asynsets):
    for pos in asynsets.keys():
        for offset in asynsets[pos].keys():
            if not "db-synset" in asynsets[pos][offset]: continue
            db_synsets = _retrieve_similar_synset(WN.synset(asynsets[pos][offset]["synset"]))
            asynsets[pos][offset]["srpwords"] = _get_srpword_from_synsets(db_synsets) #[{"srp-word": "test", "srp-pos": "noun"}] #
    n = pd.DataFrame(asynsets['noun']).T
    a = pd.DataFrame(asynsets['adj']).T
    r = pd.DataFrame(asynsets['adv']).T
    v = pd.DataFrame(asynsets['verb']).T
    s = pd.concat([n, a, r, v], axis=0)
    
    b = [k for k, v in NRC_WNA_MAPPING.items() for i in a if i in v]
    s['emo_categ'] = s['categ'].apply(lambda x: ','.join([k for k, v in NRC_WNA_MAPPING.items() if x in v]))
    s.to_csv(AFF_SYNSETS_PATH_SR.replace('xml', 'csv'), sep='\t', index=False)

    output_srp_asynset(asynsets)
    return s

# Retrieve similar synsets from WordNet
def _retrieve_similar_synset(synset):
    if not synset: return []
    similar_db_synsets = [str("%08d-%s" % (synset.offset(), synset.pos()))]
    searched_words = {}

    synsets = [synset]
    while synsets:
        for synset in synsets:
            searched_words[synset.name()] = 1

        nexts = []
        for synset in synsets:
            for syn in _get_similar_synsets(synset):
                if not syn.name() in searched_words:
                    similar_db_synsets.append(str("%08d-%s" % (syn.offset(), syn.pos())))
                    nexts.append(syn)
        synsets = nexts

    return similar_db_synsets

# Get hyponyms, similar, verb groups, entailment, pertainym
def _get_similar_synsets(synset):
    synsets = []
    synsets.append(synset.hyponyms())
    synsets.append(synset.similar_tos())
    synsets.append(synset.verb_groups())
    synsets.append(synset.entailments())
    for lemma in synset.lemmas():
        synsets.append([x.synset() for x in lemma.pertainyms()])

    return list(set(reduce(lambda x,y: x+y, synsets)))

# Get serbian word from serbian wordnet
def _get_srpword_from_synsets(synsets):
    tree = ET.parse(SWN_PATH)
    root = tree.getroot()

    rows = []
    for s in synsets: 
        synset = root.find('SYNSET/[ID="ENG30-%s"]' % (s.replace('-r', '-b').replace('-s', '-a')))
        if synset is not None:
            row = {}
            pos = synset.find('POS').text

            positive = synset.find('SENTIMENT/POSITIVE').text
            negative = synset.find('SENTIMENT/NEGATIVE').text
            domain = ''
            if synset.find('DOMAIN') != None:
                domain = synset.find('DOMAIN').text
            gloss = ''
            if synset.find('DEF') != None:
                gloss = synset.find('DEF').text

            for literal in synset.findall('SYNONYM/LITERAL'):
                print(literal.text)
                row = {
                    'srp-word': literal.text,
                    'srp-pos': pos,
                    'srp-db-synset': s,
                    'srp-gloss': gloss,
                    'srp-domain': domain,
                    'srp-positive': positive,
                    'srp-negative': negative
                }
                rows.append(row)
        else:
            continue
    return rows

# Output Serbian wordnet affect
def output_srp_asynset(asynsets):
    root = Element('syn-list')
    for pos in asynsets.keys():
        pos_node = SubElement(root, "%s-syn-list" % (pos))
        for offset, asynset in asynsets[pos].items():
            node = SubElement(pos_node, "%s-syn" % (pos))
            for attr in ["offset30", "synset", "categ", "caus-stat", "noun-synset", "srp-word"]:
                if attr in asynset:
                    node.set(attr, asynset[attr])
            if "srpwords" in asynset:
                for word in asynset["srpwords"]:
                    word_node = SubElement(node, "srp-word", {
                        "lemma": word['srp-word'],
                        "pos": word['srp-pos'],
                        'gloss': word['srp-gloss'],
                        'domain': word['srp-domain'],
                        'positive': word['srp-positive'],
                        'negative': word['srp-negative']
                    })

    file = open(AFF_SYNSETS_PATH_SR, "wb")
    file.write(minidom.parseString(tostring(root)).toprettyxml(encoding='utf-8'))
    file.close()


if __name__ == '__main__':
    asynsets_16 = load_asynsets(AFF_SYNSETS_PATH_EN)
    asynsets_30 = merge_asynset_with_wn(asynsets_16)
    asynsets_with_srp = merge_asynset_with_wnsrp(asynsets_30)
    output_srp_wnet(asynsets_with_srp, SWN_PATH)