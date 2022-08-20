import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

PALETTE = [
    'lightcoral', 'lightskyblue', 'gold', 'sandybrown', 'navajowhite', 'khaki',
    'lightslategrey', 'turquoise', 'rosybrown', 'thistle', 'pink'
]
sns.set_palette(PALETTE)
BACKCOLOR = '#f6f5f5'


def cat_dist(data, var: str, hue: str, msg_show=True, img_show: bool = True):
    total_cnt = data[var].count()
    f, ax = plt.subplots(1, 2, figsize=(25, 8))
    hues = [None, hue]
    titles = [f"{var}'s distribution", f"{var}'s distribution by {hue}"]

    for i in range(2):
        sns.countplot(data[var],
                      edgecolor='black',
                      hue=hues[i],
                      linewidth=1,
                      ax=ax[i],
                      data=data)
        ax[i].set_xlabel(var, weight='bold', size=13)
        ax[i].set_ylabel('Count', weight='bold', size=13)
        ax[i].set_facecolor(BACKCOLOR)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_title(titles[i], size=15, weight='bold')
        for patch in ax[i].patches:
            x, height, width = patch.get_x(), patch.get_height(
            ), patch.get_width()
            if msg_show:
                ax[i].text(x + width / 2,
                           height + 3,
                           f'{height} \n({height / total_cnt * 100:2.2f}%)',
                           va='center',
                           ha='center',
                           size=12,
                           bbox={
                               'facecolor': 'white',
                               'boxstyle': 'round'
                           })
    plt.savefig(f'{var}_dist.png')
    if img_show:
        plt.show()


def continuous_dist(data, x: str, y: str, img_show: bool = True):
    """ How y is distributed in x
    """
    f, ax = plt.subplots(1, 4, figsize=(35, 10))
    sns.histplot(data=data, x=y, hue=x, ax=ax[0], element='step')
    sns.violinplot(x=data[x],
                   y=data[y],
                   ax=ax[1],
                   edgecolor='black',
                   linewidth=1)
    sns.boxplot(x=data[x], y=data[y], ax=ax[2])
    sns.stripplot(x=data[x], y=data[y], ax=ax[3])
    for i in range(4):
        for e in ['top', 'right']:
            ax[i].spines[e].set_visible(False)
        ax[i].set_xlabel(x, weight='bold', size=20)
        ax[i].set_ylabel(y, weight='bold', size=20)
        ax[i].set_facecolor(BACKCOLOR)
    f.suptitle(f"{y}'s distribution by {x}", weight='bold', size=25)
    plt.savefig(f'{y}_dist_by_{x}.png')
    if img_show:
        plt.show()
