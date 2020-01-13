import re
import string
import itertools
from time import time

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from collections import defaultdict

import numpy as np
import csv
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

import gensim
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from cleaning_txt import cleaning
from baseline_model import baseline_model
from oversampling import OverSampling
from train_test_split import train_test_split
from classifier import	model_checker
from load_glove_model import load_glove_model
from tweet_to_vector import tweet_to_vector
from get_tweet_vectors import get_tweet_vectors
from word2vec import Word2Vec_Model

def main():
	
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
	
	baseline_model(df) #68.51%
	
	word_cnt = CountVectorizer(
		stop_words='english',
		strip_accents='unicode',
		analyzer='word',
		token_pattern=r'\w{1,}',
		ngram_range=(1, 1), 
		max_features=10000)
	TFIDF = TfidfVectorizer(
		strip_accents='unicode',
		analyzer='word',
		token_pattern=r'\w{1,}',
		ngram_range=(1, 1),
		max_features=10000)

	lr = LogisticRegression(solver ='liblinear',penalty='l1')
	RF = RandomForestClassifier(n_estimators = 10,bootstrap=False,max_features = 'sqrt',criterion = 'entropy', random_state=100)
	NB = MultinomialNB()
	SVM = svm.SVC(kernel = 'linear', probability = True, random_state = 100)

	
	model_checker(df,word_cnt,lr,'None') #Accuracy:76.86%  Train-Test-Split-Time:1.45s
	model_checker(df,word_cnt,lr,'Oversampling') #75.72% 0.75s

	model_checker(df,TFIDF,lr,'None') #77.09% 0.78s
	model_checker(df,TFIDF,lr,'Oversampling') #75.3% 1.35s

	model_checker(df,word_cnt,RF,'None') #75.19 4.12s
	model_checker(df,word_cnt,RF,'Oversampling') #74.24% 3.02s

	model_checker(df,TFIDF,RF,'None') #75.87% 2.46s
	model_checker(df,TFIDF,RF,'Oversampling') #74.81 2.66s

	model_checker(df,word_cnt,NB,'None') #75.42 0.32s
	model_checker(df,word_cnt,NB,'Oversampling') #70.60% 0.35s

	model_checker(df,TFIDF,NB,'None') #73.07% 0.31s
	model_checker(df,TFIDF,NB,'Oversampling') #70.68% 0.37s

	model_checker(df,word_cnt,SVM,'None') #74.85% 86.87s
	model_checker(df,word_cnt,SVM,'Oversampling') #74.73% 156.54s

	model_checker(df,TFIDF,SVM,'None') #77.01% 76.67s
	model_checker(df,TFIDF,SVM,'Oversampling') #73.18% 137.49s
	
	#https://nlp.stanford.edu/projects/glove/
	
	global GloveModel
	GloveModel = load_glove_model("glove.twitter.27B.100d.txt")  

	count_total = 0   # Number of words in original tweet including duplicated words
	count_in = 0      # Number of words in Glove pre-trained da
	count_out = 0     # Number of words are not in Glove pretrained data
	out_words_list = []    # A list of words that are not found in Glove pretrained data

	Word2Vec_Model(df,lr,'None') #74.24% 5.19s
	Word2Vec_Model(df,RF,'None') #71.62% 6.77s
	Word2Vec_Model(df,SVM,'None') #74.20% 216.76s

if __name__ == '__main__':
	main()

