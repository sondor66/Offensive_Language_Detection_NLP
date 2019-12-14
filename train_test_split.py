
from oversampling import OverSampling
from sklearn.utils import resample
from sklearn.utils import shuffle
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def train_test_split(df,sampling_method):
    
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state =42)

    for train_index,test_index in split.split(df,df['label']):
        strat_train_set = df.iloc[train_index]
        strat_test_set = df.iloc[test_index]

    if sampling_method =='Oversampling':
        strat_train_set = OverSampling(strat_train_set)

    trainTweet=strat_train_set['text']
    testTweet = strat_test_set['text']
    trainLabel=strat_train_set['label']
    testLabel=strat_test_set['label']

    return trainTweet,testTweet,trainLabel,testLabel
