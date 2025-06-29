import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from constants import EMO_CATEGORIES, CORPUS

if __name__ == '__main__':

    for crp in CORPUS:
        crp_df = pd.read_csv(f'./data/{crp}-Emo.SR.csv', sep='\t')
        
        crp_df['emotion'] = crp_df.label.apply(lambda x: [y.strip() for y in x.strip('[]').split(',') if y.strip()
                                                            in EMO_CATEGORIES]) 

        mlb = MultiLabelBinarizer()
        #s1 = ann['ann_1_1.0'].str.split(',').apply(lambda x: [y.strip(' ') for y in x])
        crp_emo_df = pd.DataFrame(mlb.fit_transform(crp_df.emotion), columns=mlb.classes_, index=crp_df.emotion.index)

        f = crp_emo_df.T.dot(crp_emo_df).rank(axis=1, method='dense', pct=True).round(3)
        g = crp_emo_df.value_counts(normalize=True)

        labelfreqs = crp_emo_df.sum(axis=0)
        examples = crp_emo_df.shape[0]
        labels = crp_emo_df.shape[1]
        print(crp)
        print('============')
        print("Examples: ", examples)
        print("labels: ", labels)
        print("Labelsets: ", crp_emo_df.astype(str).agg("".join, axis=1).nunique())
        print("Diversity: ", np.round(crp_emo_df.astype(str).agg("".join, axis=1).nunique()/np.power(2, labels), 3))
        print("Label frequencies: ", '\n', labelfreqs)
        print("MeanIR: ", np.round(np.mean(labelfreqs.max() / labelfreqs), 3))
        print("MaxIR: ", np.round(np.max(labelfreqs.max() / labelfreqs), 3))
        print("P_min: ", np.round((crp_emo_df.sum(axis=1) == 1).sum() / examples, 3))
        print("Card: ", np.round(crp_emo_df.sum().sum() / examples, 3))
        print("Dens: ", np.round(crp_emo_df.sum().sum() / examples / labels, 3))
        print('\n')

    