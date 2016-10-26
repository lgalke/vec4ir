# from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.neighbors.base import NeighborsBase, KNeighborsMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, Pipeline
from scipy.sparse import vstack
import numpy as np


def ExactMatch(Xq, X):
    # FIXME the following line only works for same shapes
    # TODO think about doing this for any query term individually O_o
    Xq_nz, other1 = np.nonzero(Xq)
    X_nz, other2 = np.nonzero(X)
    print(Xq_nz)
    print(other2)
    matching_elements = np.logical_and(Xq, X)
    row_mask = matching_elements.any(axis=1)
    return row_mask


class RetrievalModel(NeighborsBase, KNeighborsMixin, TransformerMixin):
    # TODO think about hiding KNeighborsMixin somehow, replace with
    # delegate NearestNeighbor thingy

    def __init__(self, vectorizer=TfidfVectorizer(), metric='cosine',
                 algorithm='brute', match_fn=ExactMatch, **kwargs):
        """ initializes vectorizer and passes remaining params down to
        NeighborsBase
        """
        self.vectorizer = vectorizer
        self.match_fn = ExactMatch
        self._init_params(metric=metric,
                          algorithm=algorithm,
                          **kwargs)  # NeighborsBase

    def fit(self, X, y=None):
        """ Fit the vectorizer and transform X to setup the index,
        if y is given, copy it and return its corresponding values
        on later queries. Consider y as the documents' ids """
        if y is not None and len(X) != len(y):
            raise ValueError("If you provide y make sure its length equals\
                              X.shape[0]")
        print(y)
        self._X = self.vectorizer.fit_transform(X)
        self._y = np.array(y) if y is not None else None
        return self

    def partial_fit(self, X, y=None):
        """ Add some objects into the index """
        if self._y is not None:
            if y is not None and len(X) != len(y):
                raise ValueError("Shapes...")
            self._y = vstack([self._y, y])
        Xt = self.vectorizer.transform(X)
        self._X = vstack([self._X, Xt])

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
        Xquery = self.transform(X)
        # 'matching' operation TODO make this variable
        # match_fn :: Xquery -> _X -> indices for X (and y)
        if self.match_fn is not None:
            row_mask = self.match_fn(Xquery, self._X)
            Xm, ym = self._X[row_mask], self._y[row_mask]
        else:
            Xm, ym = self._X, self._y

        self._fit(Xm)  # NeighborsBase
        n_ret = max(Xm.shape[0], k)  # dont retrieve more than available
        ind = self.kneighbors(Xquery, n_neighbors=n_ret, return_distance=False)
        if self._y is not None:
            labels = np.empty_like(ind)
            for row, i in enumerate(ind):
                labels[row, :] = ym[i]
            return labels
        else:
            return ind


def TfIdfRetrieval():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("cosine", NearestNeighbors(metric="cosine", algorithm="brute"))
    ])
