from sklearn.base import BaseEstimator, TransformerMixin
try:
    from .utils import filter_vocab
    from .core import EmbeddedVectorizer, embed
except SystemError:
    from utils import filter_vocab
    from core import EmbeddedVectorizer, embed
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.special import expit
from collections import Counter
import scipy.sparse as sp
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


class EmbeddingBasedQueryLanguageModels(BaseEstimator, TransformerMixin):
    """Embedding-based Query Language Models by Zamani and Croft 2016
    >>> sents = ["obama speaks to the press in illinois",\
                "the president talks to the media in chicago"]
    >>> ls = lambda s: s.lower().split()
    >>> from gensim.models import Word2Vec
    >>> wv = Word2Vec(sentences=[ls(sent) for sent in sents], min_count=1)
    >>> eqlm = EmbeddingBasedQueryLanguageModels(wv, analyzer=ls, m=1, eqe=2, verbose=1)
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
        self.vocabulary = {word: index for index, word in
                           enumerate(embedding.index2word)}

    def fit(self, raw_docs, y=None):
        """ Learns how to expand query with respect to corpus X """
        E = self._embedding.syn0
        a, c = self._a, self._c
        D = delta(E, E, n_jobs=self.n_jobs, a=a, c=c)
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


class CentroidExpansion(BaseEstimator):

    def __init__(self, embedding, analyzer, m=10, verbose=0,
                 **neighbor_params):
        self.embedding = embedding
        self.m = m
        self.neighbors = NearestNeighbors(n_neighbors=m, **neighbor_params)
        self.common = None
        self.vect = CountVectorizer(analyzer=analyzer,
                                    vocabulary=embedding.index2word)
        BaseEstimator.__init__(self)

    def fit(self, docs, y=None):
        """ Fit effective vocab even if embedding contains more words """
        index2word = self.embedding.index2word
        syn0 = self.embedding.syn0

        # find unique words
        X_tmp = self.vect.fit_transform(docs)
        __, cols, __ = sp.find(X_tmp)

        features = np.unique(cols)
        print('features:', features)

        # reduce vocabulary and vectors
        common_words = np.array([index2word[f] for f in features])
        common_vectors = syn0[features]
        print('CE: common vectors shape', common_vectors.shape)

        # fit nearest neighbors with vectors
        self.common = common_words
        self.neighbors.fit(common_vectors)

        return self

    def transform(self, query, y=None):
        """ Expands query by nearest tokens from collection """
        E, vect = self.embedding, self.vect
        qt = vect.transform([query])
        emb = embed(qt, E.syn0)

        ind = self.neighbors.kneighbors(emb, return_distance=False,
                                        n_neighbors=self.m)

        exp_words = self.common[ind.ravel()]

        expanded_query = query + ' ' + ' '.join(exp_words)
        print("Expanded query:", expanded_query)

        return expanded_query

    def fit_transform(self, X, y):
        raise NotImplemented('fit_transform does not make sense for expansion')


if __name__ == '__main__':
    import doctest
    doctest.testmod()
