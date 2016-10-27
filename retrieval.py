# from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.neighbors.base import NeighborsBase, KNeighborsMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, Pipeline
from scipy.sparse import vstack
import scipy.sparse as sp
import numpy as np


def TermMatch(q, X):
    indices = np.unique(X.transpose()[q.nonzero()[1], :].nonzero()[1])
    print(indices)
    return indices


class RetrievalModel(NeighborsBase, KNeighborsMixin, TransformerMixin):
    # TODO think about hiding KNeighborsMixin somehow, replace with
    # delegate NearestNeighbor thingy

    def __init__(self, vectorizer=TfidfVectorizer(), metric='cosine',
                 algorithm='brute', match_fn=TermMatch, **kwargs):
        """ initializes vectorizer and passes remaining params down to
        NeighborsBase
        """
        self.vectorizer = vectorizer
        self.match_fn = match_fn
        self._init_params(metric=metric,
                          algorithm=algorithm,
                          **kwargs)  # NeighborsBase

    def fit(self, X, y=None):
        """ Fit the vectorizer and transform X to setup the index,
        if y is given, copy it and return its corresponding values
        on later queries. Consider y as the documents' ids """
        if y is not None and len(X) != len(y):
            raise ValueError("If you provide y make sure its length equals X")
        self._X = self.vectorizer.fit_transform(X)
        self._y = np.array(y) if y is not None else None
        return self

    def partial_fit(self, X, y=None):
        """ Add some objects into the index """
        if self._y is not None:
            if y is not None and len(X) != len(y):
                raise ValueError("Shapes...")
            self._y = sp.vstack([self._y, y])
        Xt = self.vectorizer.transform(X)
        self._X = sp.vstack([self._X, Xt])

        return self

    def transform(self, X):
        return self.vectorizer.transform(X)

    def inverse_transform(self, X):
        return self.vectorizer.inverse_transform(X)

    def index(self, X, y=None):
        if self._X is None:
            self.fit(X, y)
        else:
            self.partial_fit(X, y)

    def query(self, X, k=1, **kwargs):
        print("Query started:", X)
        print("self._X.shape:", self._X.shape)
        if self._y is not None:
            print("self._y.shape:", self._y.shape)
        Xquery = self.transform(X)
        print("Xquery.shape:", Xquery.shape)
        # 'matching' operation TODO make this variable
        # match_fn :: Xquery -> _X -> indices for X (and y)
        for x in Xquery:
            print("x.shape:", x.shape)
            if self.match_fn is not None:
                row_mask = self.match_fn(x, self._X)
                if self._y is not None:
                    Xm, ym = self._X[row_mask], self._y[row_mask]
                else:
                    Xm = self._X[row_mask]
            else:
                Xm, ym = self._X, self._y

            print("Xm.shape:", Xm.shape)
            if self._y is not None:
                print("ym.shape:", ym.shape)

            self._fit(Xm)  # NeighborsBase
            n_ret = min(Xm.shape[0], k)  # dont retrieve more than available
            ind = self.kneighbors(x, n_neighbors=n_ret, return_distance=False)
            if self._y is not None:
                labels = np.choose(ind, ym)
                yield labels
            else:
                yield ind


def TfIdfRetrieval():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("cosine", NearestNeighbors(metric="cosine", algorithm="brute"))
    ])
