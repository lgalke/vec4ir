# from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.neighbors.base import NeighborsBase, KNeighborsMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import scipy.sparse as sp
import numpy as np
import gist.rank_metrics as rm
from sklearn.metrics import f1_score

def InverseTermMatch(q, X):
    # X (terms, documents)
    # q (terms, 1)
    terms = q.nonzero()[0]
    indices = np.unique(X[terms, :].nonzero()[1])
    return indices

def TermMatch(q, X):
    # X (documents, terms)
    # q (1, terms)
    indices = np.unique(X.transpose()[q.nonzero()[1], :].nonzero()[1])
    return indices

def _checkXy(X, y):
    if y is not None and len(X) != len(y):
        raise ValueError("Shapes of X and y do not match.")


def average_ndcg_at_k(rs, k, method=0):
    return np.mean([rm.ndcg_at_k(r, k, method) for r in rs])

VALID_METRICS = {"mean_reciprocal_rank": rm.mean_reciprocal_rank,
                 "mean_average_precision": rm.mean_average_precision,
                 "average_ndcg_at_k": average_ndcg_at_k}


class RetrievalModel(NeighborsBase, KNeighborsMixin, TransformerMixin):
    # TODO think about hiding KNeighborsMixin somehow, replace with
    # delegate NearestNeighbor thingy

    def __init__(self, vectorizer=TfidfVectorizer(), metric='cosine',
                 algorithm='brute', match_fn=TermMatch, verbose=0, **kwargs):
        """ initializes vectorizer and passes remaining params down to
        NeighborsBase
        """
        self.vectorizer = vectorizer
        self.match_fn = match_fn
        self.verbose = verbose
        self._init_params(metric=metric,
                          algorithm=algorithm,
                          **kwargs)  # NeighborsBase

    def fit(self, X, y=None):
        """ Fit the vectorizer and transform X to setup the index,
        if y is given, copy it and return its corresponding values
        on later queries. Consider y as the documents' ids """
        _checkXy(X, y)
        self._X = self.vectorizer.fit_transform(X)
        if y is None:
            n_docs = self._X.shape[0]
            self._y = np.arange(n_docs)
        else:
            self._y = np.asarray(y)
        return self

    def partial_fit(self, X, y=None):
        """ Add some objects into the index """
        _checkXy(X, y)
        Xt = self.vectorizer.transform(X)
        self._X = sp.vstack([self._X, Xt])
        if y is None:
            next_id = np.amax(self._y) + 1
            new_ids = np.arange(next_id, next_id + Xt.shape[0])
            self._y = np.hstack([self._y, new_ids])
        else:
            self._y = np.hstack(self._y, np.asarray(y))

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
        results = []
        for x in Xquery:
            if self.match_fn is not None:
                row_mask = self.match_fn(x, self._X)
                Xm, ym = self._X[row_mask], self._y[row_mask]
            else:
                Xm, ym = self._X, self._y
            self._fit(Xm)  # NeighborsBase
            n_ret = min(Xm.shape[0], k)  # dont retrieve more than available
            ind = self.kneighbors(x, n_neighbors=n_ret, return_distance=False)
            labels = np.choose(ind, ym)
            results.append(labels.ravel())
        return np.asarray(results)

    def score(self, X, Y, k=20, metrics=VALID_METRICS.keys()):
        """ Y: relevancy table of shape (n_queries, n_samples) with relevancy\
        values"""
        assert Y.shape == (len(X), self._X.shape[0])
        rs = []
        for qid, result in enumerate(self.query(X, k)):
            r = Y[qid, result]
            rs.append(r)

        rs = np.asarray(rs)
        print("DEBUG rs:", rs)
        values = {}
        if "average_ndcg_at_k" in metrics:
            values["average_ndcg_at_k"] = average_ndcg_at_k(rs, k)
        if "mean_reciprocal_rank" in metrics:
            values["mean_reciprocal_rank"] = rm.mean_reciprocal_rank(rs)
        if "mean_average_precision" in metrics:
            values["mean_average_precision"] = rm.mean_average_precision(rs)

        # print(rs)
        # nz = np.count_nonzero(rs)
        # y = sp.csr_matrix((np.ones()), shape=Y.shape)
        # y[rs] = 1
        # print(y)
        # values["f1_score"] = f1_score(Y, y)

        return values


def TfIdfRetrieval():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("cosine", NearestNeighbors(metric="cosine", algorithm="brute"))
    ])
