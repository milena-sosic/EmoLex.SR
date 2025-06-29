import pandas as pd
from utils import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import cohen_kappa_score, jaccard_score
import krippendorff
import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

def compute_fleiss_kappa_per_label(Y):
    n_annotators = len(Y)
    n_samples, n_labels = Y[0].shape
    kappas = {}
    for label_idx in range(n_labels):
        counts_matrix = np.zeros((n_samples, 2), dtype=int)
        for sample_idx in range(n_samples):
            votes = [Y[ann][sample_idx, label_idx] for ann in range(n_annotators)]
            count_ones = sum(votes)
            counts_matrix[sample_idx, 1] = count_ones
            counts_matrix[sample_idx, 0] = n_annotators - count_ones
        kappas[label_idx] = fleiss_kappa(counts_matrix)
    return kappas

def compute_agreement(annotations: List[str]):

    dfs = [pd.read_csv(f, sep='\t') for f in annotations]
    for df in dfs:
        df['label_final'] = df['label_final'].fillna('')

    assert all(len(df) == len(dfs[0]) for df in dfs), "Annotation files must have same number of rows."

    def to_label_list(series):
        return series.apply(lambda x: x.split('|') if isinstance(x, str) else [])

    label_lists = [to_label_list(df['label_final']) for df in dfs]
    mlb = MultiLabelBinarizer()
    all_labels = pd.concat(label_lists)
    mlb.fit(all_labels)
    Y = [mlb.transform(lbls) for lbls in label_lists]

    print("Cohen's Kappa Score per label:")
    for i, label in enumerate(mlb.classes_):
        k12 = cohen_kappa_score(Y[0][:, i], Y[1][:, i])
        k13 = cohen_kappa_score(Y[0][:, i], Y[2][:, i])
        k23 = cohen_kappa_score(Y[1][:, i], Y[2][:, i])
        print(f"{label:12}: k12={k12:.2f}, k13={k13:.2f}, k23={k23:.2f}, avg={np.mean([k12, k13, k23]):.2f}")

    print("Krippendorff's Alpha per label:")
    for i, label in enumerate(mlb.classes_):
        data = np.vstack([Y[0][:, i], Y[1][:, i], Y[2][:, i]])
        alpha = krippendorff.alpha(reliability_data=data, level_of_measurement='nominal')
        print(f"{label:12}: alpha={alpha:.2f}")

    print("Average Jaccard Similarity:")
    j12 = np.mean([jaccard_score(Y[0][i], Y[1][i]) for i in range(len(Y[0]))])
    j13 = np.mean([jaccard_score(Y[0][i], Y[2][i]) for i in range(len(Y[0]))])
    j23 = np.mean([jaccard_score(Y[1][i], Y[2][i]) for i in range(len(Y[1]))])
    print(f"Jaccard 1↔2: {j12:.2f}, 1↔3: {j13:.2f}, 2↔3: {j23:.2f}")

    print("Fleiss' Kappa per label:")
    fleiss_scores = compute_fleiss_kappa_per_label(Y)
    for i, label in enumerate(mlb.classes_):
        print(f"{label:12}: fleiss_kappa={fleiss_scores[i]:.2f}")


if __name__ == '__main__':
    compute_agreement('./lexicons/manual_annotation/annotator_1.csv', './lexicons/manual_annotation/annotator_2.csv', './lexicons/manual_annotation/annotator_3.csv')
    