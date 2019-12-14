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
	
	baseline_model(df)
	
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

	
	model_checker(word_cnt,lr,'None')
	model_checker(word_cnt,lr,'Oversampling')

	model_checker(TFIDF,lr,'None')
	model_checker(TFIDF,lr,'Oversampling')

	model_checker(word_cnt,RF,'None')
	model_checker(word_cnt,RF,'Oversampling')

	model_checker(TFIDF,RF,'None')
	model_checker(TFIDF,RF,'Oversampling')

	model_checker(word_cnt,NB,'None')
	model_checker(word_cnt,NB,'Oversampling')

	model_checker(TFIDF,NB,'None')
	model_checker(TFIDF,NB,'Oversampling')

	model_checker(word_cnt,SVM,'None')
	model_checker(wword_cnt,SVM,'Oversampling')

	model_checker(TFIDF,SVM,'None')
	model_checker(TFIDF,SVM,'Oversampling')
	
	#https://nlp.stanford.edu/projects/glove/
	
	global GloveModel
	GloveModel = load_glove_model("glove.twitter.27B.100d.txt")  

	count_total = 0   # Number of words in original tweet including duplicated words
	count_in = 0      # Number of words in Glove pre-trained da
	count_out = 0     # Number of words are not in Glove pretrained data
	out_words_list = []    # A list of words that are not found in Glove pretrained data

	Word2Vec_Model(df,lr,'None')
	Word2Vec_Model(df,RF,'None')
	Word2Vec_Model(df,SVM,'None')

if __name__ == '__main__':
	main()

