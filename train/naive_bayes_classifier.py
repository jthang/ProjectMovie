import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer

def make_xy(critics, vectorizer=None):
    if vectorizer is None:
        vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(critics.quote)
    X = X.tocsc()
    Y = (critics.fresh == 'fresh').values.astype(np.int)
    return X, Y

def log_likelihood(clf, x, y):
    prob = clf.predict_log_proba(x)
    rotten = y == 0
    fresh = ~rotten
    return prob[rotten, 0].sum() + prob[fresh, 1].sum()

def cv_score(clf, x, y, score_func):
    result = 0
    nfold = 5
    for train, test in KFold(y.size, nfold):
        clf.fit(x[train], y[train])
        result += score_func(clf, x[test], y[test]) 
    return result / nfold

def maximize_cv(cv_score, log_likelihood):
    alphas = [0, .1, 1, 5, 10, 50]
    min_dfs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    best_alpha = None
    best_min_df = None
    max_loglike = -np.inf

    for alpha in alphas:
        for min_df in min_dfs:         
            vectorizer = CountVectorizer(min_df = min_df)       
            X, Y = make_xy(critics, vectorizer)
                        
            X_training,xtest,Y_training,ytest = train_test_split(X,Y)
            clf = MultinomialNB(alpha=alpha)
            
            temp = cv_score(clf, xtest, ytest,log_likelihood)
            if temp > max_loglike:
                best_alpha = alpha
                best_min_df = min_df
                max_loglike = temp
    
    return best_alpha, best_min_df

if __name__ == '__main__':
    
    movies = pd.read_csv('../data/movies.dat', delimiter='\t')
    critics = pd.read_csv('../data/critics.csv')

    critics = critics[~critics.quote.isnull()]
    critics = critics[critics.fresh != 'none']
    critics = critics[critics.quote.str.len() > 0]

    df =  critics.groupby('critic')
    X, Y = make_xy(critics)
    X_training, xtest, Y_training, ytest = train_test_split(X,Y)

    clf = MultinomialNB()
    clf.fit(X_training,Y_training)

    test_score = clf.score(xtest,ytest)
    training_score = clf.score(X_training, Y_training)

    log_likelihood(clf, xtest, ytest)
    alpha, min_df = maximize_cv(cv_score, log_likelihood)

    vectorizer2 = CountVectorizer(min_df = min_df)
    quotes = critics.quote.tolist()
    vectorizer2.fit(quotes)
    X2 = vectorizer2.transform(quotes)
    Y2 = [int(i=='fresh') for i in critics.fresh]
    X2 = X2.toarray()
    X_train2,xtest2,Y_train2,ytest2 = train_test_split(X2,Y2)
    clf2 = MultinomialNB(alpha = alpha)
    clf2.fit(xtest2, ytest2)

    words = vectorizer2.get_feature_names()

    num_rows = len(words)   
    eye = np.eye(num_rows)
    eye_prob = clf2.predict_proba(eye)
    eye_prob = eye_prob.tolist()
    eye_fresh = [k[1] for k in eye_prob]
    eye_rot = [k[0] for k in eye_prob]

    df_words= pd.DataFrame({'words': words, 'fresh_prob': eye_fresh, 'eye_rot': eye_rot})
    df_words = df_words.sort_index(by='fresh_prob')

    fresh_predict = df_words.tail(10)
    rot_predict = df_words.head(10)

    print 'Words most predictive of fresh reviews: ' + str(df_words.words[-10:])
    print 'Words most predictive of rotten reviews: ' + str(df_words.words[:10])








    