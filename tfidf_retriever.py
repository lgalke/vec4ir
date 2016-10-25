# from sklearn.base import BaseEstimator
from sklearn.neighbors import BallTree
from sklearn.text import TfifdVectorizer


class TfidfRetriever(object):
    def __init__(self, name="TFIDF", vect=TfifdVectorizer(), leaf_size=128):
        self.name = name
        self.vectorizer = vect
        self.leaf_size = leaf_size

    def fit(self, X):
        _X = self.vectorizer.fit_transform(X)
        self.tree = BallTree(_X, metric='cosine', leaf_size=self.leaf_size)

    def transform(self, X):
        return self.vectorizer.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def query(self, X):
        _X = self.transform(X)
        # matching?
        indices = self.tree.query(_X, sort_results=True)
        return indices

    def __str__(self):
        return self.name
