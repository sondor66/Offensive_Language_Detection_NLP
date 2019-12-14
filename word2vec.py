from get_tweet_vectors import get_tweet_vectors
from train_test_split import train_test_split
from load_glove_model import load_glove_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from time import time 
import pandas as pd
import numpy as np 

def Word2Vec_Model(df, classifier,sampling_method):
    print(classifier)
    print('\n')
    GloveModel = load_glove_model("glove.twitter.27B.100d.txt")  

    trainTweet,testTweet,trainLabel,testLabel = train_test_split(df,sampling_method)        
    pipeline = Pipeline([('classifier',classifier)])
    
    global count_total, count_in, count_out
    global out_words_list
    count_total, count_in, count_out = 0, 0, 0 
    out_words_list = []    
    
    trainVec = get_tweet_vectors(trainTweet, GloveModel, 100) # it has to be same as read in txt dimension which is 200.
    testVec = get_tweet_vectors(testTweet, GloveModel, 100) # glove.twitter.27B.200d.txt
    
    print("Glove word embedding statistic\n", "count_total: %d/" %count_total, "count_in: %d/" %count_in, "count_out: %d/" %count_out)
    print("Number of unique words without embedding: %d" %len(set(out_words_list)))
    print("Words without embedding: \n", set(out_words_list))
    
    t0 = time() 
    pipeline.fit(trainVec,trainLabel)  
    y_pred = pipeline.predict(testVec)
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