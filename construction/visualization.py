import pandas as pd
from pyplutchik import plutchik
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from load_lexicons import *


def plot_emo_words_pyplutchik(min_categories=0):

    df = load_sr_lexicon(version=2)
    df = df[df.label != 'neutral']
    df['label_len'] = df.label.str.split('|').apply(len)
    df = df[df.label_len > min_categories]
    
    df = df.sample(9).reset_index(drop=True) 
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))
    
    for i, row in df.iterrows():
        ax_i = plt.subplot(3, 3, i + 1)
        emotions_simple = row[['anger','anticipation','disgust','fear','joy','sadness','surprise','trust']].to_dict()
        plutchik(emotions_simple, title=row['lemma'], ax=ax_i, show_coordinates=True, title_size = 12, fontweight = 'normal', fontsize = 8, normalize=0.4)
    plt.tight_layout(pad=2)
    
    plt.savefig('./images/emo_words_pyplutchik.png')
    plt.show()


def draw_pie_plot(df_agg):
    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{:.1f}%\n({v:d})'.format(pct, v=val)

        return my_format

    df_agg_plot = df_agg.assign(label_plot=df_agg['label'].str.split('|')).explode('label_plot').reset_index(drop=True)
    emotions = ['joy', 'neutral',
                'trust',
                'fear',
                'surprise',
                'sadness',
                'disgust',
                'anger',
                'anticipation']
    s = df_agg_plot['label_plot'].value_counts()[emotions]
    print(s)
    ct = pd.crosstab(df_agg_plot['label_plot'], df_agg_plot['pos'])
    print(ct)
    ct.to_csv('emotion_distribution2.csv', sep='\t', index=False)

    SMALL_SIZE = 9
    plt.rc('font', size=SMALL_SIZE)
    texts = plt.pie(s, labels=s.index, autopct=autopct_format(s), pctdistance=0.85, radius=1.3,
    # Add space around each slice
    explode=[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                    colors=[
                        'gold', 'silver', 'olivedrab', 'forestgreen', 'skyblue', 'dodgerblue', 'slateblue', 'orangered',
                        'darkorange'], wedgeprops={"alpha": 0.6}
            ) 
    for text in texts[2]:
        text.set_horizontalalignment('center')

    plt.show()
    