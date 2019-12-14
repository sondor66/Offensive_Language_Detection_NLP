import re
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from collections import defaultdict

def cleaning(txt):

	cleanReview = ''
	review = word_tokenize(txt)
	stopword = []

	for i in review:
		#replace consecutive non-ASCII characters with a space
		i = re.sub(r'[^\x00-\x7F]+',' ',i)
    	#removes punctuation 
    	#i.translate(str.maketrans('', '', string.punctuation)) 
		# remove digits
		i = re.sub(r'\d+', '', i)
		# remove html tags
		i = re.sub('(?:<[^>]+>)', '',i)
		#remove more than one space
		i = re.sub(r"\s+","", i)
		#lower case first letter and keep all uppercase word
		if i.isupper():
			i = i
		else:
			i = i.lower()
		stopword = stopwords.words('english')
		stopword = list(set(stopword))
		Extra = ["@USER","USER","URL",".",";",":","/","\\",",","#","@","$","&",")","(","\""]
		stopword = stopword + Extra
		if i not in stopword:
			cleanReview = cleanReview + ' ' + i 
		#	print(cleanReview)
	return cleanReview
