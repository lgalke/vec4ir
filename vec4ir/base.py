#!/usr/bin/env python3

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import maxabs_scale
from abc import abstractmethod
from collections import defaultdict, OrderedDict
from operator import itemgetter
from operator import mul
from functools import reduce
from timeit import default_timer as timer
import scipy.sparse as sp
import numpy as np

try:
    from . import rank_metrics as rm
except SystemError:
    import rank_metrics as rm


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
    # print("matching matching_doc_indices", matching_doc_indices,
    # file=sys.stderr)
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
    >>> ind = retrieval._matching( "fox" )
    >>> print(ind.shape)
    (1,)
    >>> str(docs[ind[0]])
    'the quick brown fox'
    >>> ind
    array([0], dtype=int32)
    >>> len(retrieval._matching( "brown dog" ))
    2
    """
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    def _init_params(self, name=None, match_fn='term', **kwargs):
        # reasonable defaults for indexing use case
        binary = kwargs.pop('binary', True)
        dtype = kwargs.pop('dtype', np.bool_)
        self._match_fn = TermMatch if match_fn == 'term' else match_fn
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

    def _matching(self, query):
        match_fn = self._match_fn
        _X = self._inv_X
        q = self._cv.transform(np.asarray([query]))
        # q = self._cv.transform(query)
        ind = match_fn(_X, q)
        return ind


class RetriEvalMixIn():
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def query(X, k=1):
        pass

    def evaluate(self, X, Y, k=20, verbose=0, replacement="zero"):
        """
        X : [(qid, str)] query id, query pairs
        Y : pandas dataseries with qid,docid index
        """
        rs = []
        tpq = []
        ndcgs = []
        for qid, query in X:
            # execute query
            if verbose > 0:
                print(qid, ":", query)
            t0 = timer()
            result = self.query(query)
            tpq.append(timer() - t0)
            # result = result[:k]  # TRIM HERE
            # soak the generator
            scored_result = []
            for docid in result:
                # could surround with try-catch to support dict of dicts or nmpy
                score = Y.get((qid, docid), None)
                if score is not None:
                    scored_result.append(score)
                elif replacement:
                    scored_result.append(replacement)
                if len(scored_result) == k:
                    break
            # replacement with relevancy values
            # if verbose:
            #     for docid in result:
            #         print(docid)
            # r = [Y.loc(axis=0)[qid, docid] for docid in result]
            # print(result)
            # try:
            #     ranks = [Y.get((qid, docid), 0) for docid in result]
            # except AttributeError:
            #     ranks = [Y[qid][docid] for docid in result]

            # padding does not change scores
            # r += [0] * (k - len(r))  # python magic for padding
            if verbose > 0:
                print(scored_result)
            idcg = rm.dcg_at_k(sorted(Y.get(qid)), k)
            ndcgs.append(rm.ndcg_at_k(scored_result, k) / idcg)
            rs.append(scored_result)
        values = {}
        # values["ndcg_at_k"] = np.asarray([rm.ndcg_at_k(r, k) for r in rs])
        values["ndcg_at_k"] = ndcgs
        # values["precision@5"] = np.asarray([rm.precision_at_k(r, 5)
        #                                     for r in rs])
        # values["precision@10"] = np.asarray([rm.precision_at_k(r, 10)
        #                                      for r in rs])
        values["mean_reciprocal_rank"] = rm.mean_reciprocal_rank(rs)
        values["mean_average_precision"] = rm.mean_average_precision(rs)
        values["time_per_query"] = np.mean(np.asarray(tpq)) * 1000
        return values


def aggregate_dicts(dicts, agg_fn=sum):
    """
    Aggregates the contents of two dictionaries by key
    @param agg_fn is used to aggregate the values (defaults to sum)
    >>> dict1 = {'a': 0.8, 'b': 0.4, 'd': 0.4}
    >>> dict2 = {'a': 0.7, 'c': 0.3, 'd': 0.3}
    >>> agg = aggregate_dicts([dict1, dict2])
    >>> OrderedDict(sorted(agg.items(), key=itemgetter(1), reverse=True))
    OrderedDict([('a', 1.5), ('d', 0.7), ('b', 0.4), ('c', 0.3)])
    """
    acc = defaultdict(list)
    for d in dicts:
        for k in d:
            acc[k].append(d[k])

    for key, values in acc.items():
        acc[key] = agg_fn(values)

    return dict(acc)  # no need to default to list anymore


def product(values):
    """
    Computes the product from a list of values
    >>> product([2,3,2])
    12
    >>> product([0,2,3])
    0
    >>> product([42])
    42
    """
    return reduce(mul, values)


def fuzzy_or(values):
    """
    Applies fuzzy-or to a list of values
    >>> fuzzy_or([0.5])
    0.5
    >>> fuzzy_or([0.5, 0.5])
    0.75
    >>> fuzzy_or([0.5, 0.5, 0.5])
    0.875
    """
    if min(values) < 0 or max(values) > 0:
        raise ValueError("fuzzy_or expects values in [0,1]")
    return reduce(lambda x, y: 1 - (1 - x) * (1 - y), values)


class CombinatorMixIn(object):
    """ Creates a computational tree with retrieval models as leafs
    """
    def __get_weights(self, other):
        if not isinstance(other, CombinatorMixIn):
            raise ValueError("other is not Combinable")

        if hasattr(self, '__weight'):
            weight = self.__weight
        else:
            weight = 1.0

        if hasattr(other, '__weight'):
            otherweight = other.__weight
        else:
            otherweight = 1.0

        return weight, otherweight

    # This is evil since it can exceed [0,1], rescaling at the end would be not
    # that beautiful
    # def __add__(self, other):
    #     weights = self.__get_weights(other)
    #     return Combined([self, other], weights=weights, agg_fn=sum)

    def __and__(self, other):
        weights = self.__get_weights(other)
        return Combined([self, other], weights=weights, agg_fn=product)

    def __or__(self, other):
        weights = self.__get_weights(other)
        return Combined([self, other], weights=weights, agg_fn=fuzzy_or)

    def __mul__(self, scalar):
        self.__weight = scalar
        return self


class Combined(BaseEstimator, CombinatorMixIn):
    def __init__(self, retrieval_models, weights=None, aggregation_fn=sum):
        self.retrieval_models = retrieval_models
        self.aggregation_fn = aggregation_fn
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1.0] * len(retrieval_models)

    def query(self, query, k=1, sort=True):
        models = self.retrieval_models
        weights = maxabs_scale(self.weights)  # max 1 does not crash [0,1]
        agg_fn = self.aggregation_fn
        # we only need to sort in the final run
        combined = [m.query(query, k=k, sort=False) for m in models]

        if weights is not None:
            combined = [{k: v * w for k, v in r.items()} for r, w in
                        zip(combined, weights)]

        combined = aggregate_dicts(combined, agg_fn=agg_fn, sort=True)

        if sort:
            # only cut-off at k if this is the final (sorted) output
            combined = OrderedDict(sorted(combined.items(), key=itemgetter(1),
                                          reverse=True)[:k])
        return combined


class TfidfRetrieval(RetrievalBase, CombinatorMixIn, RetriEvalMixIn):
    """
    Class for tfidf based retrieval
    >>> tfidf = TfidfRetrieval(input='content')
    >>> docs = ["The quick", "brown fox", "jumps over", "the lazy dog"]
    >>> _ = tfidf.fit(docs)
    >>> tfidf._y.shape
    (4,)
    >>> values = tfidf.evaluate(zip([0,1],["fox","dog"]),\
    >>> [{0:0,1:1,2:0,3:0}, {0:0,1:0,2:0,3:1}], k=20)
    >>> import pprint
    >>> pprint.pprint(values)
    {'mean_average_precision': 1.0,
     'mean_reciprocal_rank': 1.0,
     'ndcg_at_k': array([ 1.,  1.])}
    >>> _ = tfidf.partial_fit(["new fox doc"])
    >>> list(tfidf.query("new fox doc",k=2))
    [4, 1]
    >>> values = tfidf.evaluate([(0,"new fox doc")], np.asarray([[0,2,0,0,0]]), k=3)
    >>> pprint.pprint(values)
    {'mean_average_precision': 0.5,
     'mean_reciprocal_rank': 0.5,
     'ndcg_at_k': array([ 1.])}
    """

    def __init__(self, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False, **kwargs):
        self.tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
                                      smooth_idf=smooth_idf,
                                      sublinear_tf=sublinear_tf)

        # override defaults since we need the counts here
        self.verbose = kwargs.get('verbose', 0)

        binary = kwargs.pop('binary', False)
        dtype = kwargs.pop('dtype', np.int64)

        # pass remaining args to countvectorizer
        self._init_params(name="TFIDF", binary=binary, dtype=dtype, **kwargs)

    def fit(self, X, y=None):
        self._fit(X, y)  # set _inv_X for matching
        self.tfidf.fit(self._inv_X)  # fit idf on _inv_X (counts are stored)
        self._X = self.tfidf.transform(self._inv_X)  # transform X
        return self

    def partial_fit(self, X, y=None):
        self._partial_fit(X, y)
        Xt = self.tfidf.transform(self._cv.transform(X))
        self._X = sp.vstack([self._X, Xt])
        return self

    def query(self, query):
        # matching step
        matching_ind = self._matching(query)
        # print(matching_ind, file=sys.stderr)
        Xm, matched_doc_ids = self._X[matching_ind], self._y[matching_ind]
        # matching_docs, matching_doc_ids = self._matching(query)
        # calculate elements to retrieve
        n_match = len(matching_ind)
        if n_match == 0:
            return []
        if self.verbose > 0:
            print("Found {} matches:".format(n_match))
        # n_ret = min(n_match, k) if k > 0 else n_match
        # model dependent transformation
        xq = self._cv.transform([query])
        q = self.tfidf.transform(xq)
        # Xm = self.vectorizer.transform(matching_docs)
        # model dependent nearest neighbor search or scoring or whatever
        nn = NearestNeighbors(metric='cosine', algorithm='brute').fit(Xm)
        # abuse kneighbors in this case
        # AS q only contains one element, we only need its results.
        ind = nn.kneighbors(q,  # q contains a single element
                            n_neighbors=n_match,  # limit to k neighbors
                            return_distance=False)[0]  # so we only need 1 res
        # dont forget to convert the indices to document ids of matching
        labels = matched_doc_ids[ind]
        return labels


if __name__ == '__main__':
    import doctest
    doctest.testmod()
