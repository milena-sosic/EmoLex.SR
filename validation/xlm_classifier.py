import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.special import softmax
from sklearn.metrics import classification_report, accuracy_score
from constants import *
from lexicon_validation.metrics import measure_accuracy
import pickle

with open(RELDIA_SR_LEX, 'rb') as file:
    lexicon = pickle.load(file)

model_emo_labels = ['anger', 'fear', 'joy', 'sadness']

class XLMEmotionClassifier:
    def __init__(self, corpus_path):
        self.df = pd.read_csv(corpus_path, sep='\t')
        self.df['label'] = self.df['label'].apply(lambda x: ','.join([y for y in x.split(',') if y in model_emo_labels]))
        self.df = self.df[self.df.label != '']
    
        self.model = AutoModelForSequenceClassification.from_pretrained(XLM_MODEL_PATH, resume_download=True)
        self.tokenizer = AutoTokenizer.from_pretrained(XLM_MODEL_PATH, resume_download=True)
        self.config = AutoConfig.from_pretrained(XLM_MODEL_PATH)
        self.mlb = MultiLabelBinarizer()

    def predict_emotion(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt') 
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return scores

    def predict_xlm_labels(self, text, level=1):
        row = {}
        scores = []
        scores_list = []
        if len(text) > 512:
            while len(text) > 0:
                scores = self.predict_emotion(text[:511])
                scores_list.append(scores)
                text = text[511:]
        else:
            scores = self.predict_emotion(text)
        if scores_list:
            scores = np.asarray(scores_list).max(axis=0, keepdims=False)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        
        for i in range(scores.shape[0]):
            l = self.config.id2label[i]
            row[l] = scores[i]
        result = []
        for i in range(level):
            result.append(self.config.id2label[ranking[i]])
        result = ','.join(result)
        return result

    def preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def train_and_evaluate(self):

        print(f"\n=== XLM Prediction ===")
        self.Y = self.mlb.fit_transform(self.df['label'].str.split(','))

        for i in range(1, 4):
            Y_pred = self.df['sentence_sr'].apply(lambda x: self.predict_xlm_labels(self.preprocess(x), level=i))
            self.df[f'labels_v{i}'] = Y_pred

            Y_pred_trans = self.mlb.transform(Y_pred.str.split(','))
            print("Overall accuracy:", accuracy_score(self.Y, Y_pred_trans))
            print("Detailed per-label metrics:\n")
            print(classification_report(self.Y, Y_pred_trans, target_names=self.mlb.classes_))

        self.df['gold_label'] = self.df['label']

        measure_accuracy(self.df, type='corpus-label')


def main():
    clf = XLMEmotionClassifier("./data/LLM-Emo.SR.csv")

    clf.train_and_evaluate()

if __name__ == "__main__":
    main()
