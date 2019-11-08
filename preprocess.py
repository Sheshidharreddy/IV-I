import anvil.server

anvil.server.connect("6I75POTUMBQSOLRITQ3Y42RO-TVTAGRSEHECEEAOJ") 

    
X = bbc['article']

import numpy as np
from nltk.corpus import stopwords 
sw = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(stop_words = sw, min_df = 2)

X = vect.fit_transform(X)


import scipy

X = scipy.sparse.csr_matrix.todense(X)
X = np.array(X)


import numpy as np
Y = np.array(bbc['category'])
    
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Y = le.fit_transform(Y)
print(le.transform(['business','entertainment','politics','sport','tech']))


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.75)

print(le.transform(['business','entertainment','politics','sport','tech']))

@anvil.server.callable
def pre():  
    return dict(zip(['business','entertainment','politics','sport','tech'], le.transform(['business','entertainment','politics','sport','tech'])))

@anvil.server.callable
def tst():
    return Y_test
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
