#!/usr/bin/env python
# coding: utf-8
"""
File: base.py
Author: Lukas Galke
Email: vim@lpag.de
Github: https://github.com/lgalke
Description: Base classed for (embedding-based) retrieval.
"""

from timeit import default_timer as timer

from abc import abstractmethod
from collections import defaultdict
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import NotFittedError
from joblib import Parallel, delayed
import scipy.sparse as sp
import numpy as np
import pandas as pd

try:
    from . import rank_metrics as rm
    from .utils import argtopk
    from .combination import CombinatorMixin
except (SystemError, ValueError):
    from combination import CombinatorMixin
    from utils import argtopk
    import rank_metrics as rm


def f1_score(precision, recall):
    """
    Computes the harmonic mean of precision and recall (f1 score)
    """
    if precision == 0 and recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def harvest(source, query_id, doc_id=None, default=0):
    """ harvest source for either a sorted list of relevancy scores for a given
    query id or a relevance score for a queryid, docid pair)

    Arguments:
        source -- {pandas data frame, list of dicts, ndarray}
        query_id -- the query id to harvest the answers for

    Keyword Arguments:
        doc_id -- if None, return sorted relevancy scores for query with
        query_id
        default -- default value if no relevance score is available in source

    >>> ll = [[2,3,4,5],[22,33,42,55]]
    >>> harvest(ll, 1, 2)
    42
    >>> harvest(ll, 1, -42, 1337)
    1337
    >>> sorted(harvest(ll, 1), reverse=True)
    [55, 42, 33, 22]
    >>> nda = np.array(ll)
    >>> harvest(nda, 1, 2)
    42
    >>> harvest(nda, 1, -42, 1337)
    1337
    >>> ld = [{"d1":2,"d2":3,"d3":4,"d4":5},{"d1":22,"d2":33,"d3":42,"d4":55}]
    >>> harvest(ld, 1, "d3")
    42
    >>> harvest(ld, 1, "fail", 1337)
    1337
    >>> sorted(harvest(ld, 0))
    [2, 3, 4, 5]
    """
    is_pd = isinstance(source, pd.Series)
    is_dict = isinstance(source, dict)
    if doc_id is None:
        # Return sorted list of relevance scores for that query
        if is_pd or is_dict:
            # source is pandas series or dict
            scores = source.get(query_id)
        else:
            # source is ndarray or list
            scores = source[query_id]

#         try:
#             # scores is numpy array-like?
#             scores = np.sort(scores)[::-1]
#         except ValueError:
#             # probably scores is a dict itself...
        if isinstance(scores, dict):
            scores = np.asarray(list(scores.values()))
        else:
            scores = np.asarray(scores)
#             scores = np.sort(scores)[::-1]
        return scores
    else:
        # Return relevance score for the respective (query, document) pair
        # try:  # pandas multi index df
        if is_pd:
            score = source.get((query_id, doc_id), default)
        else:
            # default dict or ndarray
            scores = source[query_id]
            # no special treatment for ndarray since we want to raise exception
            # when query id is out of bounds
            try:
                # ok now if scores provides a get
                # (pandas or dict), lets use it:
                score = scores.get(doc_id, default)
            except AttributeError:
                # try the brute force access
                try:
                    score = scores[doc_id]
                # the user should be aware when he wants to index his stuff
                # by numpy indexing: only int doc ids allowed, of course.
                except IndexError:
                    score = default

        return score


def filter_none(some_list):
    """ Just filters None elements out of a list """
    old_len = len(some_list)
    new_list = [l for l in some_list if l is not None]
    diff = old_len - len(new_list)
    return new_list, diff


def pad(r, k, padding=0):
    """ pads relevance scores with zeros to given length """
    r += [padding] * (k - len(r))  # python magic for padding
    return r


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


def _checkXy(X, y):
    if y is not None and len(X) != len(y):
        raise ValueError("Shapes of X and y do not match.")


class Matching(BaseEstimator):

    """Typical Matching Operation of Retrieval Systems"""

    def __init__(self, match_fn=TermMatch, binary=True, dtype=np.bool_,
                 **cv_params):
        """initializes a Matching object

        :match_fn: A matching function of signature `docs, query`
                    -> indices of matching docs
        :binary: Store only binary term occurrences.
        :dtype: Data type of internal feature matrix
        :cv_params: Parameter for the count vectorizer such as lowercase=True

        """
        # RetrievalBase.__init__(self)

        self._match_fn = match_fn
        self._vect = CountVectorizer(binary=binary, dtype=dtype,
                                     **cv_params)

    def fit(self, X):
        cv = self._vect

        self._fit_X = cv.fit_transform(X)  # fit internal countvectorizer

        return self

    def predict(self, query):
        cv, match_fn, fit_X = self._vect, self._match_fn, self._fit_X
        # q = cv.transform(np.asarray([query]))
        q = cv.transform(([query]))
        ind = match_fn(fit_X, q)
        return ind


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
        self._init_params(**kwargs)

    def _init_params(self,
                     name=None,
                     match_fn='term',
                     binary=True,
                     dtype=np.bool_,
                     **kwargs):
        # reasonable defaults for indexing use case
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

    def fit(self, X, y=None):
        return self._fit(X, y)

    def _partial_fit(self, X, y=None):
        _checkXy(X, y)
        # update index
        self._inv_X = sp.vstack([self._inv_X, self._cv.transform(X)])
        # update source
        # self._fit_X = np.hstack([self._fit_X, np.asarray(X)])
        # try to infer viable doc ids
        if y is None:
            next_id = np.amax(self._y) + 1
            y = np.arange(next_id, next_id + len(X))
        else:
            y = np.asarray(y)
        self._y = np.hstack([self._y, y])

        self.n_docs += len(X)
        return self

    def partial_fit(self, X, y=None):
        self._partial_fit(X, y)

    def _matching(self, query):
        match_fn = self._match_fn
        _X = self._inv_X
        q = self._cv.transform(np.asarray([query]))
        # q = self._cv.transform(query)
        ind = match_fn(_X, q)
        return ind


def process_and_evaluate(model, X, Y, k, n_jobs=1):
    """
    Arguments:
        X : query_id, query pairs
        Y : dict of dicts (harvestable)
        k : int how many to retrieve

    """
    print("Starting query time with %d jobs" % n_jobs)

    # TODO can we unzip Y and only pass the fucking chunk of y which 
    # it needs to harvest??
    qids_rs = Parallel(n_jobs=n_jobs)(delayed(process_query)(model, x, Y, k)
                                      for x in X)

    print("Evaluating the results:")

    scores = evaluate_results(qids_rs, Y, k)

    return scores


def process_query(model, x, Y, k):
    """ Processes one query, good for parallelization
    :x: pair of query_id, query
    :Y: dict of dict - like of relevance with query_id's as key in outer dict
    and doc_ids as keys of inner dict
    """
    qid, query = x
    # t0 = timer()
    print("{} : {}", qid, query)
    result = model.query(query, k=k)
    if k is not None:
        result = result[:k]
    # values["time_per_query"].append(timer() - t0)
    relevance = [harvest(Y, qid, docid) for docid in result]
    return (qid, np.asarray(relevance))


def evaluate_results(qids_rs, Y, k):
    values = defaultdict(list)
    for qid, r in qids_rs:
        gold = harvest(Y, qid)
        gold_topk = gold[argtopk(gold, k)]
        R = np.count_nonzero(gold_topk)
        # real ndcg
        idcg = rm.dcg_at_k(gold_topk, k)
        ndcg = rm.dcg_at_k(r, k) / idcg
        values["ndcg"].append(ndcg)
        # Verified

        # MAP@k
        ap = rm.average_precision(r)
        values["MAP"].append(ap)

        # MRR - compute by hand
        ind = np.asarray(r).nonzero()[0]
        mrr = (1. / (ind[0] + 1)) if ind.size else 0.
        values["MRR"].append(mrr)

        # R precision
        # R = min(R, k)  # ok lets be fair.. you cant get more than k
        # we dont need that anymore, since we chop of the remainder
        # before computing R
        recall = rm.recall(r, R)
        values["recall"].append(recall)

        # precision = rm.precision_at_k(pad(scored_result, k), k)
        precision = rm.precision(r)
        values["precision"].append(precision)

        f1 = f1_score(precision, recall)
        values["f1_score"].append(f1)

        # Safe variant does not fail if len(r) < k
        p_at_5 = rm.safe_precision_at_k(r, 5)
        values["precision@5"].append(p_at_5)

        p_at_10 = rm.safe_precision_at_k(r, 10)
        values["precision@10"].append(p_at_10)
    return values


class RetriEvalMixin():

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def query(X, k=None):
        pass

    def evaluate(self, X, Y, k=20, verbose=0, replacement=0, n_jobs=1):
        """
        :X: [(qid, str)] query id, query string pairs
        :Y: pandas dataseries with qid,docid index or [dict]
        :k: Limit the result for all metrics to this value, the models are also
        given a hint of how many they should return.
        :replacement: 0 means that (query, doc) pairs not prevalent in Y will
        not be considered relevant, None means that those are not considered
        (skipped).
        """
        # rs = []

        # if n_jobs > 1:
        #     return process_and_evaluate(self, X, Y, k, n_jobs)
        values = defaultdict(list)
        for qid, query in X:
            # execute query
            if verbose > 0:
                print(qid, ":", query)
            t0 = timer()
            # if replacement is None, we need to drop after querying
            result = self.query(query, k=(None if replacement is None else k))
            values["time_per_query"].append(timer() - t0)
            # if verbose > 0:
            #     print(result[:k])
            # result = result[:k]  # TRIM HERE
            # soak the generator
            scored_result = [harvest(Y, qid, docid, replacement)
                             for docid in result]
            if replacement is None:
                scored_result, notfound = filter_none(scored_result)
                values["gold_not_found"].append(notfound)

            if k is not None:
                # dont let the models cheat by returning more than k
                r = scored_result[:k]
            else:
                # if k is None, consider all
                r = scored_result

            # if verbose > 0:
            #     print(r)

            # gold = np.array(list(Y[qid].values()))
            gold = harvest(Y, qid)
            import sys
            # print(gold, file=sys.stderr)
            topk_indices = argtopk(gold, k)
            print(topk_indices, file=sys.stderr)
            gold_topk = gold[topk_indices]
            # print('Top k in gold standard:', gold_topk, file=sys.stderr)
            R = np.count_nonzero(gold_topk)
            if verbose > 0:
                print("Retrieved {} relevant out of {} possible."
                      .format(np.count_nonzero(r), R))

            # real ndcg
            idcg = rm.dcg_at_k(gold_topk, k)
            ndcg = rm.dcg_at_k(scored_result, k) / idcg
            values["ndcg"].append(ndcg)
            # Verified

            # MAP@k
            ap = rm.average_precision(r)
            values["MAP"].append(ap)

            # MRR - compute by hand
            ind = np.asarray(r).nonzero()[0]
            mrr = (1. / (ind[0] + 1)) if ind.size else 0.
            values["MRR"].append(mrr)

            # R precision
            # R = min(R, k)  # ok lets be fair.. you cant get more than k
            # we dont need that anymore, since we chop of the remainder
            # before computing R
            recall = rm.recall(r, R)
            values["recall"].append(recall)

            # precision = rm.precision_at_k(pad(scored_result, k), k)
            precision = rm.precision(r)
            values["precision"].append(precision)

            f1 = f1_score(precision, recall)
            values["f1_score"].append(f1)

            # Safe variant does not fail if len(r) < k
            p_at_5 = rm.safe_precision_at_k(r, 5)
            values["precision@5"].append(p_at_5)

            p_at_10 = rm.safe_precision_at_k(r, 10)
            values["precision@10"].append(p_at_10)

            # rs.append(r)
            if verbose > 0:
                # print("Precision: {:.4f}".format(precision))
                # print("Recall: {:.4f}".format(recall))
                # print("F1-Score: {:.4f}".format(f1))
                print("AP: {:.4f}".format(ap))
                print("RR: {:.4f}".format(mrr))
                print("NDCG: {:.4f}".format(ndcg))

        return values


class Tfidf(TfidfVectorizer):
    def __init__(self, analyzer='word', use_idf=True):
        TfidfVectorizer.__init__(self, analyzer=analyzer, use_idf=use_idf,
                                 norm='l2')
        self._fit_X = None

    def fit(self, X):
        Xt = super().fit_transform(X)
        self._fit_X = Xt
        return self

    def query(self, query, k=None, indices=None, return_scores=False):
        if self._fit_X is None:
            raise NotFittedError
        q = super().transform([query])
        if indices is not None:
            fit_X = self._fit_X[indices]
        else:
            fit_X = self._fit_X
        # both fit_X and q are l2-normalized
        D = linear_kernel(q, fit_X)
        ind = argtopk(D[0], k)
        if return_scores:
            return ind, D[0,ind]
        else:
            return ind

class TfidfRetrieval(RetrievalBase, CombinatorMixin, RetriEvalMixin):
    """
    Class for tfidf based retrieval
    >>> tfidf = TfidfRetrieval(input='content')
    >>> docs = ["The quick", "brown fox", "jumps over", "the lazy dog"]
    >>> _ = tfidf.fit(docs)
    >>> tfidf._y.shape
    (4,)
    >>> gold = [{0:0,1:1,2:0,3:0}, {0:0,1:0,2:0,3:1}]
    >>> values = tfidf.evaluate(zip([0,1],["fox","dog"]), gold, k=20)
    >>> import pprint
    >>> pprint.pprint(values["MAP"])
    [1.0, 1.0]
    >>> _ = tfidf.partial_fit(["new fox doc"])
    >>> list(tfidf.query("new fox doc"))
    [4, 1]
    >>> gold = np.asarray([[0,2,0,0,0]])
    >>> values = tfidf.evaluate([(0,"new fox doc")], gold, k=3)
    >>> pprint.pprint(values["MAP"])
    [0.5]
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

    def query(self, query, k=None, matched_indices=None):
        # matching step
        matching_ind = self._matching(query)
        # print(matching_ind, file=sys.stderr)
        Xm, matched_doc_ids = self._X[matching_ind], self._y[matching_ind]
        # matching_docs, matching_doc_ids = self._matching(query)
        # calculate elements to retrieve
        n_ret = len(matching_ind)
        if n_ret == 0:
            return []
        if self.verbose > 0:
            print("Found {} matches:".format(n_ret))
        # n_ret = min(n_ret, k) if k > 0 else n_ret
        # model dependent transformation
        xq = self._cv.transform([query])
        q = self.tfidf.transform(xq)
        # Xm = self.vectorizer.transform(matching_docs)
        # model dependent nearest neighbor search or scoring or whatever
        nn = NearestNeighbors(metric='cosine', algorithm='brute').fit(Xm)
        # abuse kneighbors in this case
        # AS q only contains one element, we only need its results.
        if k is not None and k < n_ret:
            n_ret = k

        ind = nn.kneighbors(q,  # q contains a single element
                            n_neighbors=n_ret,  # limit to k neighbors
                            return_distance=False)[0]  # so we only need 1 res
        # dont forget to convert the indices to document ids of matching
        labels = matched_doc_ids[ind]
        return labels


if __name__ == '__main__':
    import doctest
    doctest.testmod()
