# get word2vec vector for each tweet        
import numpy as np
from tweet_to_vector import tweet_to_vector
from load_glove_model import load_glove_model
def get_tweet_vectors(tweets, GloveModel, num_features): 

	GloveModel = load_glove_model("glove.twitter.27B.100d.txt")  

	curr_ind = 0
	tweet_feature_vecs = np.zeros((len(tweets), num_features), dtype = "float32")

	for tweet in tweets:
	    if curr_ind % 2000 == 0:
	        print('Word2vec vectorizing tweet %d of %d' %(curr_ind, len(tweets)))
	    tweet_feature_vecs[curr_ind] = tweet_to_vector(tweet, GloveModel, num_features)
	    curr_ind += 1

	return tweet_feature_vecs   