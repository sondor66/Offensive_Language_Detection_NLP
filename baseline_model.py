from textblob import TextBlob
# %config InlineBackend.figure_format = 'retina'
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd

def baseline_model(df):
	tbresult = [TextBlob(i).sentiment.polarity for i in df['text']]
	tbpred = [1 if n<0 else 0 for n in tbresult]

	result = confusion_matrix(tbpred,df['label'])
	print ('Baseline Model:\n')
	print ("Accuracy Score: {0:.2f}%".format(accuracy_score(df['label'], tbpred)*100))
	print ("-"*80)
	print ("Confusion Matrix\n")
	print (pd.DataFrame(result))
	print ("-"*80)
	print ("Classification Report\n")
	print (classification_report(df['label'], tbpred))