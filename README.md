# Offensive_Language_Detection_NLP
The dataset OLID(The Offensive Language Identification Dataset) is from https://sites.google.com/site/offensevalsharedtask/olid. It contains 14,200 annotated English tweets using an annotation model that encompasses 3 levels. The first level which is offensive language detection is used in this project.

# The workflow of this file:
    1. Read data 'olidtrain.csv'.
    2. Data manipulation(cleaning texts, remove duplicates, change column names etc.)
    3. Data exploration is in a seperate jupyter notebook file in a seperate repository - 'NLP_Offensive_Speech_Exploratory_Analysis'.
    4. Feature Engineering: TF/TFIDF/Word2Vec Embeddings (Glove)
    5. Models (Logistic Regression, Naive Bayes, Random Forest, Support Vector Machine)
    6. Deep Learning (Deep Neural Network, Convolutional Neural Network, LSTM). It's in a seperate Jupyter Notebook -               'Traditional_VS_DeepLearning'
