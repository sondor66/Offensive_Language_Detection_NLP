from train_test_split import train_test_split
from cleaning_txt import cleaning

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from time import time 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def model_checker(vectorizer,classifier,sampling_method):

    print(classifier)
    print('\n')

    df_train = pd.read_csv('olidtrain.csv')
    #print(df_train.head())
    #get rid of duplicated tweets
    df_train.drop_duplicates(subset = 'tweet',keep = False,inplace = True)
    df = df_train[['tweet','subtask_a']]
    # change the label to 1 and 0
    df['subtask_a'] = (df['subtask_a'] == 'OFF').astype(int)
    #rename the column name
    df.columns = ['text', 'label']
    # cleaning text
    df['text'] = df['text'].apply(lambda x :cleaning(x)) 
    
    trainTweet,testTweet,trainLabel,testLabel = train_test_split(df,sampling_method)
    
    #vectorizer = vectorizer.fit_transform(trainLabel.values.astype('U'))
    pipeline = Pipeline([('vectorizer',vectorizer),
                               ('classifier',classifier)])
    #vectorizer = vectorizer.fit_transform(trainTweet.apply(lambda x: np.str_(x)))
    t0 = time() 
    sentiment_fit = classifier.fit(trainTweet,trainLabel)  #trainTweet
    y_pred = sentiment_fit.predict(testTweet)
    train_test_time = time() - t0
    
    accuracy = accuracy_score(testLabel,y_pred)
    confusion_result = confusion_matrix(y_pred,testLabel)
    
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    print("train and test time: {0:.2f}s".format(train_test_time))
    print('-'*80)
    print ("Confusion Matrix\n")
    print (pd.DataFrame(confusion_result))
    print('-'*80)
    print ("Classification Report\n")
    print (classification_report(testLabel,y_pred))

    
