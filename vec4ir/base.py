#!/usr/bin/env python3

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from abc import abstractmethod
import scipy.sparse as sp
import numpy as np

try:
    from . import rank_metrics as rm
except SystemError:
    import rank_metrics as rm
import sys

VALID_METRICS = ["mean_reciprocal_rank", "mean_average_precision", "average_ndcg_at_k"]


def TermMatch(X, q):
    """
    X : ndarray of shape (documents, terms)
    q : ndarray of shape (1, terms)
    >>> X = np.array([[0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0]])
    >>> TermMatch(X, np.array([[0,0,0]]))
    array([], dtype=int64)
    >>> TermMatch(X, np.array([[0,0,1]]))
    array([0, 2, 4])
    >>> TermMatch(X, np.array([[0,1,0]]))
    array([1, 2, 5])
    >>> TermMatch(X, np.array([[0,1,1]]))
    array([0, 1, 2, 4, 5])
    >>> TermMatch(X, np.array([[1,0,0]]))
    array([3, 4, 5])
    >>> TermMatch(X, np.array([[1,0,1]]))
    array([0, 2, 3, 4, 5])
    >>> TermMatch(X, np.array([[1,1,0]]))
    array([1, 2, 3, 4, 5])
    >>> TermMatch(X, np.array([[1,1,1]]))
    array([0, 1, 2, 3, 4, 5])
    >>> TermMatch(X, np.array([0,1,1]))
    Traceback (most recent call last):
      File "/usr/lib64/python3.5/doctest.py", line 1320, in __run
        compileflags, 1), test.globs)
      File "<doctest __main__.TermMatch[9]>", line 1, in <module>
        TermMatch(np.array([0,1,1]), X)
      File "retrieval.py", line 50, in TermMatch
        indices = np.unique(X.transpose()[q.nonzero()[1], :].nonzero()[1])
    IndexError: tuple index out of range
    """
    # indices = np.unique(X.transpose()[q.nonzero()[1], :].nonzero()[1])
    # print("matching X", X, file=sys.stderr)
    # print("matching q", q, file=sys.stderr)
    inverted_index = X.transpose()
    # print("matching inverted_index", inverted_index, file=sys.stderr)
    query_terms = q.nonzero()[1]
    # print("matching query_terms", query_terms, file=sys.stderr)
    matching_terms = inverted_index[query_terms, :]
    # print("matching matching_terms", matching_terms, file=sys.stderr)
    matching_doc_indices = np.unique(matching_terms.nonzero()[1])
    # print("matching matching_doc_indices", matching_doc_indices, file=sys.stderr)
    return matching_doc_indices


def cosine_similarity(X, query, n_retrieve):
    """
    Computes the `n_retrieve` nearest neighbors using cosine similarity
    Xmatched : The documents that have matching terms (if matching='terms')
    q : the query
    n_retrieve : The number of indices to return.
    >>> X = np.array([[10,1,0], [1,10,0], [0,0,10]])
    >>> cosine_similarity(X, np.array([[0,23,0]]), 2)
    array([1, 0])
    >>> cosine_similarity(X, np.array([[1,0,0]]), 2)
    array([0, 1])
    >>> cosine_similarity(X, np.array([[1,0,10]]), 3)
    array([2, 0, 1])
    """
    nn = NearestNeighbors(metric='cosine', algorithm='brute').fit(X)
    ind = nn.kneighbors(query, n_neighbors=n_retrieve, return_distance=False)
    return ind.ravel()  # we want a plain list of indices


def _checkXy(X, y):
    if y is None:
        return
    if len(X) != len(y):
        raise ValueError("Shapes of X and y do not match.")


def average_ndcg_at_k(rs, k, method=1):
    """ method 0 behaves strange as it rates [1,2] as perfect """
    ndcgs = [rm.ndcg_at_k(r, k, method) for r in rs]
    return np.mean(ndcgs)


class RetrievalBase(BaseEstimator):
    """
    Provides:
    _fit_X : the source documents
    _inv_X : the (pseudo-) inverted index
    _y: the document ids
    such that _fit_X[i] ~ _inv_X[i] ~ _y[i] corresponds to each other.
    _matching(Xquery) : returns the matching subset of _fit_X
    For subclassing, the query method should return doc ids which are stored in
    _y.
    >>> retrieval = RetrievalBase()
    >>> retrieval._init_params()
    >>> docs = ["the quick brown fox", "jumps over the lazy dog"]
    >>> _ = retrieval._fit(docs, [0,1])
    >>> retrieval._inv_X.dtype
    dtype('bool')
    >>> retrieval.n_docs
    2
    >>> retrieval._inv_X.shape
    (2, 8)
    >>> retrieval._y.shape
    (2,)
    >>> ind = retrieval._matching( "fox" , return_indices=True)
    >>> print(ind.shape)
    (1,)
    >>> str(docs[ind[0]])
    'the quick brown fox'
    >>> ind
    array([0], dtype=int32)
    >>> len(retrieval._matching( "brown dog" , return_indices=True))
    2
    """
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    def _init_params(self, name=None, matching='term', **kwargs):
        # reasonable defaults for indexing use case
        binary = kwargs.pop('binary', True)
        dtype = kwargs.pop('dtype', np.bool_)
        self._match_fn = TermMatch if matching == 'term' else matching
        self._cv = CountVectorizer(binary=binary, dtype=dtype, **kwargs)
        self.name = name

    def _fit(self, X, y=None):
        """
        learn vocab and construct (pseudo-inverted) index
        """
        _checkXy(X, y)
        cv = self._cv
        self._inv_X = cv.fit_transform(X)
        # self._fit_X = np.asarray(X)
        n_docs = len(X)
        self._y = np.arange(n_docs) if y is None else np.asarray(y)
        self.n_docs = n_docs
        return self

    def _partial_fit(self, X, y=None):
        _checkXy(X, y)
        # update index
        self._inv_X = sp.vstack([self._inv_X, self._cv.transform(X)])
        # update source
        # self._fit_X = np.hstack([self._fit_X, np.asarray(X)])
        # try to infer viable doc ids
        next_id = np.amax(self._y) + 1
        if y is None:
            y = np.arange(next_id, next_id + len(X))
        else:
            y = np.asarray(y)
        self._y = np.hstack([self._y, y])

        self.n_docs += len(X)
        return self

    def _matching(self, query, return_indices=False):
        match_fn = self._match_fn
        _X = self._inv_X
        q = self._cv.transform(np.asarray([query]))
        # q = self._cv.transform(query)
        if match_fn is not None:
            ind = match_fn(_X, q)
            if return_indices is True:
                return ind
            else:
                return self._fit_X[ind], self._y[ind]
        else:
            return self._fit_X, self._y


class RetriEvalMixin():
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def query(X, k=1):
        pass

    def evaluate(self, X, Y, k=20, verbose=0, metrics=VALID_METRICS):
        """
        X : [(qid, str)] query id, query pairs
        Y : pandas dataseries with qid,docid index
        """
        rs = []
        for qid, query in X:
            # execute query
            if verbose > 0:
                print(qid, ":", query)
            result = self.query(query, k=k, verbose=verbose-1)
            # replacement with relevancy values
            # if verbose:
            #     for docid in result:
            #         print(docid)
            # r = [Y.loc(axis=0)[qid, docid] for docid in result]
            try:
                r = [Y.get((qid, docid), 0) for docid in result]
            except AttributeError:
                r = [Y[qid][docid] for docid in result]
            if verbose > 0:
                print(r)
            rs.append(r)
        values = {}
        if "average_ndcg_at_k" in metrics:
            values["average_ndcg_at_k"] = average_ndcg_at_k(rs, k)
        if "mean_reciprocal_rank" in metrics:
            values["mean_reciprocal_rank"] = rm.mean_reciprocal_rank(rs)
        if "mean_average_precision" in metrics:
            values["mean_average_precision"] = rm.mean_average_precision(rs)
        return values

    def score(self, X, Y, k=20, metrics=VALID_METRICS):
        """
        assumes a query(X,q) -> sorted_doc_ids method
        X: Query strings
        Y: relevancy values of shape (n_queries, n_samples) or [dict]
        k: number of documents to retrieve and consider in metrics
        """
        rs = []
        for qid, result in enumerate(self.query(X, k)):
            try:
                # (n_queries x n_documents)
                r = [Y[qid, docid] for docid in result]
            except TypeError:
                # [dict()]
                r = [Y[qid][docid] for docid in result]
            rs.append(r)

        # print("rs:", rs, file=sys.stderr)
        values = {}
        if "average_ndcg_at_k" in metrics:
            values["average_ndcg_at_k"] = average_ndcg_at_k(rs, k)
        if "mean_reciprocal_rank" in metrics:
            values["mean_reciprocal_rank"] = rm.mean_reciprocal_rank(rs)
        if "mean_average_precision" in metrics:
            values["mean_average_precision"] = rm.mean_average_precision(rs)

        return values


class TfidfRetrieval(RetrievalBase, RetriEvalMixin):
    """
    Class for tfidf based retrieval
    >>> tfidf = TfidfRetrieval(input='content')
    >>> docs = ["The quick", "brown fox", "jumps over", "the lazy dog"]
    >>> _ = tfidf.fit(docs)
    >>> tfidf._y.shape
    (4,)
    >>> values = tfidf.evaluate(zip([0,1],["fox","dog"]), [{0:0,1:1,2:0,3:0}, {0:0,1:0,2:0,3:1}], k=20)
    >>> import pprint
    >>> pprint.pprint(values)
    {'average_ndcg_at_k': 1.0,
     'mean_average_precision': 1.0,
     'mean_reciprocal_rank': 1.0}
    >>> _ = tfidf.partial_fit(["new fox doc"])
    >>> list(tfidf.query("new fox doc",k=2))
    [4, 1]
    >>> values = tfidf.evaluate([(0,"new fox doc")], np.asarray([[0,2,0,0,0]]), k=3)
    >>> pprint.pprint(values)
    {'average_ndcg_at_k': 0.63092975357145753,
     'mean_average_precision': 0.5,
     'mean_reciprocal_rank': 0.5}
    """

    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)
        self._init_params(name="TFIDF")

    def fit(self, X, y=None):
        self._fit(X, y)
        self.vectorizer.fit(X, y)
        self._X = self.vectorizer.transform(X)
        return self

    def partial_fit(self, X, y=None):
        self._partial_fit(X, y)
        Xt = self.vectorizer.transform(X)
        self._X = sp.vstack([self._X, Xt])
        return self

    def query(self, query, k=1, verbose=0):
        # matching step
        matching_ind = self._matching(query, return_indices=True)
        # print(matching_ind, file=sys.stderr)
        Xm, matched_doc_ids = self._X[matching_ind], self._y[matching_ind]
        # matching_docs, matching_doc_ids = self._matching(query)
        # calculate elements to retrieve
        n_match = len(matching_ind)
        if verbose > 0:
            print("Found {} matches:".format(n_match))
        n_ret = min(n_match, k)
        if not n_ret:
            return []
        # model dependent transformation
        q = self.vectorizer.transform([query])
        # Xm = self.vectorizer.transform(matching_docs)
        # model dependent nearest neighbor search or scoring or whatever
        nn = NearestNeighbors(metric='cosine', algorithm='brute').fit(Xm)
        # abuse kneighbors in this case
        # AS q only contains one element, we only need its results.
        ind = nn.kneighbors(q,  # q is a single element
                            n_neighbors=n_ret,
                            return_distance=False)[0]  # so we only need 1 res
        # dont forget to convert the indices to document ids of matching
        labels = matched_doc_ids[ind]
        return labels


if __name__ == '__main__':
    import doctest
    doctest.testmod()
