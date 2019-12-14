# get vector for each word, add vectors and take the average of the vector
import numpy as np

def tweet_to_vector(tweet, GloveModel, num_features):   

    global count_total, count_in, count_out
    count_total = 0   # Number of words in original tweet including duplicated words
    count_in = 0      # Number of words in Glove pre-trained da
    count_out = 0     # Number of words are not in Glove pretrained data
    out_words_list = [] 
    
    word_count = 0
    feature_vectors = np.zeros((num_features), dtype = "float32")
    
    for word in tweet.split(' '):
        count_total += 1
        if word in GloveModel.keys():   
            count_in += 1
            word_count += 1
            feature_vectors += GloveModel[word]
        else:
            count_out += 1
            out_words_list.append(word)

    if (word_count != 0):
        feature_vectors /= word_count

    return feature_vectors
