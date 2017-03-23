from sklearn.base import BaseEstimator
try:
    from .core import EmbeddedVectorizer
    from .utils import argtopk
except SystemError:
    from core import EmbeddedVectorizer
    from utils import argtopk
from sklearn.metrics.pairwise import pairwise_distances, linear_kernel
from sklearn.preprocessing import normalize
from scipy.special import expit
from collections import Counter
import numpy as np


def delta(X, Y, n_jobs=-1, a=1, c=0):
    """Pairwise delta function: cosine and sigmoid

    :X: TODO
    :returns: TODO

    """
    D = pairwise_distances(X, Y, metric="cosine", n_jobs=n_jobs)
    if c != 0:
        D -= c
    if a != 1:
        D *= a
    D = expit(D)
    return D


class EmbeddedQueryExpansion(BaseEstimator):
    """Embedding-based Query Language Models by Zamani and Croft 2016
    >>> sents = ["obama speaks to the press in illinois",\
                "the president talks to the media in chicago"]
    >>> ls = lambda s: s.lower().split()
    >>> from gensim.models import Word2Vec
    >>> wv = Word2Vec(sentences=[ls(sent) for sent in sents], min_count=1)
    >>> eqlm = EmbeddedQueryExpansion(wv, analyzer=ls, m=1, eqe=2, verbose=1)
    >>> _ = eqlm.fit(sents)
    >>> eqlm.transform('obama press')
    """

    def __init__(self, embedding, m=10, analyzer=None, eqe=1, verbose=0, a=1,
                 c=0, n_jobs=1):
        """
        Initializes the embedding based query language model query expansion
        technique
        """
        BaseEstimator.__init__(self)
        self._embedding = embedding
        self._analyzer = analyzer
        if eqe not in [1, 2]:
            raise ValueError
        self._eqe = eqe
        self.verbose = verbose
        self._a = a
        self._c = c
        self.m = m
        self.n_jobs = n_jobs
        self.vocabulary = None

    def fit(self, raw_docs, y=None):
        """ Learn vocabulary to index and distance matrix of words"""
        wv = self._embedding
        E = wv._embedding.wv.syn0
        a, c = self._a, self._c
        D = delta(E, E, n_jobs=self.n_jobs, a=a, c=c)
        self.vocabulary = {word: index for index, word in
                           enumerate(wv.index2word)}
        self._D = D

    def transform(self, query, y=None):
        """ Transorms a query into an expanded version of the query.
        """
        wv, D, = self._embedding, self._D
        analyze, eqe = self._analyzer, self._eqe
        vocabulary, m = self.vocabulary, self.m
        q = [vocabulary[w] for w in analyze(query)]  # [n_terms]
        c = Counter(q)
        if eqe == 1:
            prior = np.sum(D, axis=1)  # [n_words, 1] could be precomputed
            print("prior.shape", prior.shape)
            conditional = D[q] / prior  # [n_terms, n_words]
            print("conditional.shape", conditional.shape)
            posterior = prior * np.product(conditional, axis=0)  # [1, n_words]
            print("posterior.shape", posterior.shape)
            topm = np.argpartition(posterior, -m)[-m:]
            print("topm.shape", topm.shape)
            expansion = [wv.index2word[i] for i in topm]
        elif eqe == 2:
            qnorm = np.asarray([(c[i] / len(q)) for i in q])
            print("qnorm.shape", qnorm.shape)
            nom = D[:, q]
            print("nom.shape", nom.shape)
            denom = np.sum(D[q], axis=1)
            print("denom.shape", denom.shape)
            frac = nom / denom
            print("frac.shape", frac.shape)
            normfrac = frac * qnorm
            print("normfrac.shape", normfrac.shape)
            posterior = np.sum(normfrac, axis=0)
            print("posterior.shape", posterior.shape)
            topm = np.argpartition(posterior, -m)[-m:]
            print("topm.shape", topm.shape)
            expansion = [wv.index2word[i] for i in topm]
        print("Expanding:", *expansion)
        return ' '.join([query, *expansion])

    def fit_transform(self, X, y):
        raise NotImplemented('fit_transform does not make sense for expansion')


class CentroidExpansion(BaseEstimator):

    def __init__(self, embedding, analyzer='word', m=10, verbose=0,
                 use_idf=True, **ev_params):
        """Expand a query by the nearest known tokens to its centroid
        """
        self.embedding = embedding
        self.m = m
        self.vect = EmbeddedVectorizer(embedding,
                                       analyzer=analyzer,
                                       use_idf=use_idf,
                                       **ev_params)
        BaseEstimator.__init__(self)

    def fit(self, X, y=None):
        """ Fit effective vocab even if embedding contains more words """
        self.vect.fit(X)
        return self

    def transform(self, query, y=None):
        """ Expands query by nearest tokens from collection """
        wv, vect = self.embedding, self.vect

        v = vect.transform([query])[0]

        exp_tuples = wv.similar_by_vector(v, topn=self.m)
        words, __scores = zip(*exp_tuples)

        # ind = self.neighbors.kneighbors(emb, return_distance=False,
        #                                 n_neighbors=self.m)

        # exp_words = self.common[ind.ravel()]

        expanded_query = query + ' ' + ' '.join(words)
        print("Expanded query:", expanded_query)

        return expanded_query

    def fit_transform(self, X, y):
        raise NotImplemented('fit_transform does not make sense for expansion')


if __name__ == '__main__':
    import doctest
    doctest.testmod()
