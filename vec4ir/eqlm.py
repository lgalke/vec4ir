from .base import RetrievalBase, RetriEvalMixIn, TfidfRetrieval
from .utils import argtopk
from scipy.spatial.distance import cosine
from scipy.special import expit
import numpy as np


def delta(u, v):
    # TODO scale with a and c
    return expit(cosine(u, v))


def eqe1(E, query, priors):
    """
    Arguments:
        E - word embedding
        Q - list of query terms
        priors - precomputed priors (strongly recommended)
    """
    V = E.index2word

    def conditional(qi, w):
        return delta(E[qi], E[w]) / priors[w]

    posterior = np.asarray([priors[w] *
                            np.product(conditional(qi, w) for qi in query)
                            for w in V])

    return posterior


class EmbeddingBasedQueryLanguageModel(RetrievalBase, RetriEvalMixIn):

    def __init__(self,
                 embedding,
                 name="EQLM",
                 m=10,
                 eqe=1,
                 erm=False,
                 vocab_analyzer=None,
                 verbose=0,
                 **kwargs):
        if eqe not in [1, 2]:
            raise ValueError("eqe parameter must be either 1 or 2")
        self.eqe = eqe
        self.erm = erm
        self.embedding = embedding
        self.m = m
        self.name = name
        self.retrieval_model = TfidfRetrieval()
        self._init_params(**kwargs)
        if vocab_analyzer is not None:
            self.analyzer = vocab_analyzer
        else:
            self.analyzer = self._cv.build_analyzer()

    def fit(self, docs, labels):
        E = self.embedding
        V = E.index2word
        # analyze = self.analyzer
        # self._fit(docs, labels)
        self.retrieval_model.fit(docs, labels)

        # self.documents_ = np.asarray([analyze(doc) for doc in docs])

        self.priors = {w: sum(delta(E[w], E[v]) for v in V) for w in V}

    def query(self, query):
        E, m = self.embedding, self.m
        V = E.index2word
        priors = self.priors

        q = self.analyzer(query)
        posterior = eqe1(E, q, priors)
        expansion = V[argtopk(posterior, m)]

        expanded_query = " ".join(q + expansion)

        # ind = self._matching(" ".join(q))
        # D, Y = self.documents_[ind], self._y[ind]

        # compute document likelihoods
        return self.retrieval_model.query(expanded_query)
