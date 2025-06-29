import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
import re
from load_lexicons import load_lemma_pos_model
from utils import multilabel_train_test_split
from diacritics.redi import redi
import pickle
import os
from constants import *
from lexicon_validation.metrics import measure_accuracy

with open(RELDIA_SR_LEX, 'rb') as file:
    lexicon = pickle.load(file)

class MLEmotionClassifier:
    def __init__(self, corpus_path):
        self.df = pd.read_csv(corpus_path, sep='\t')
        self.df['label'] = self.df['label'].apply(lambda x: x.split(','))
        self.df['label'] = self.df['label'].apply(lambda x: [y.strip(' ') for y in x if y != 'neutral'])
        self.lemma_df = load_lemma_pos_model()
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,1), max_features=10000)
        self.mlb = MultiLabelBinarizer()

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def lemmatize_text(self, text):
        print(text)
        tokens = self.tokenize(text)
        tokens = redi(tokens, lexicon=lexicon, lm=None)
        lemma_map = self.lemma_df.drop_duplicates(subset='word').set_index('word')['lemma'].to_dict()
        lemmatized = [lemma_map.get(token, '') for token in tokens if token not in HELPER_VERBS_SR]
        lemmatized = [lemma for lemma in lemmatized if lemma != '']
        print(lemmatized)
        return ' '.join(lemmatized)
    

    def preprocess(self):
        self.df['text_lemmatized'] = self.df['sentence_sr'].apply(self.lemmatize_text)
        self.X = self.df['text_lemmatized']
        self.Y = self.mlb.fit_transform(self.df['label'])


    def train_and_evaluate(self):
        X_train, X_test, Y_train, Y_test = multilabel_train_test_split(self.X, self.Y, stratify=self.Y, test_size=0.20)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        classifiers = {
            "Logistic Regression": OneVsRestClassifier(LogisticRegression(max_iter=1000, solver='liblinear')),
            "SVM": OneVsRestClassifier(LinearSVC()),
            "Random Forest": OneVsRestClassifier(RandomForestClassifier(n_estimators=100))
        }

        for i, (name, clf) in enumerate(classifiers.items()):
            print(f"\n=== {name} ===")
            clf.fit(X_train_vec, Y_train)
            Y_pred = clf.predict(X_test_vec)

            # Convert predicted labels to list of strings separated by commas
            predicted_labels = self.mlb.inverse_transform(Y_pred)
            predicted_labels_str = [','.join(labels) for labels in predicted_labels]
            X_test = pd.DataFrame(X_test)
            X_test[f'labels_v{i+1}'] = predicted_labels_str

            print("Overall accuracy:", accuracy_score(Y_test, Y_pred))
            print("Detailed per-label metrics:\n")
            print(classification_report(Y_test, Y_pred, target_names=self.mlb.classes_, zero_division=0))

        # Convert true labels to list of strings separated by commas for comparison
        true_labels = self.mlb.inverse_transform(Y_test)
        true_labels_str = [','.join(labels) for labels in true_labels]
        X_test['gold_label'] = true_labels_str

        measure_accuracy(X_test, type='corpus-label')


def main():
    for corpus in CORPUS:
        clf = MLEmotionClassifier(f"./data/{corpus}-Emo.SR.csv")
        clf.preprocess()
        clf.train_and_evaluate()

if __name__ == "__main__":
    main()
