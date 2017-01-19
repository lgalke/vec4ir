#!/usr/bin/env python3
# coding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
# from scipy.spatial.distance import cosine
import numpy as np
try:
    from .base import RetrievalBase, RetriEvalMixIn
    # from .utils import argtopk
    from .utils import filter_vocab
    from .combination import CombinatorMixIn
except (ValueError, SystemError):
    from base import RetrievalBase, RetriEvalMixIn
    from combination import CombinatorMixIn

default_analyzer = CountVectorizer().build_analyzer()


class StringSentence(object):
    """
    Uses analyze_fn to decompose strings into words
    analyze_fn : callable to for analysis :: string -> [string]
    documents : iterable of documents
    c : the window size
    >>> documents = ["The quick brown fox jumps over the lazy dog"]
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> analyze_fn = CountVectorizer().build_analyzer()
    >>> analyze_fn(documents[0]) == documents[0].lower().split()
    True
    >>> sentences = StringSentence(documents, analyze_fn, 3)
    >>> x = list(sentences)
    >>> len(x)
    3
    >>> x[2]
    ['the', 'lazy', 'dog']
    >>> sentences = StringSentence(documents, analyze_fn, 5)
    >>> x = list(sentences)
    >>> len(x)
    2
    >>> x[0]
    ['the', 'quick', 'brown', 'fox', 'jumps']
    >>> x[1]
    ['over', 'the', 'lazy', 'dog']
    """
    def __init__(self, documents, analyze_fn=None, max_sentence_length=10000):
        if analyze_fn is None:
            self.analyze_fn = default_analyzer
        else:
            self.analyze_fn = analyze_fn
        self.documents = documents
        self.max_sentence_length = max_sentence_length

    def __iter__(self):
        for document in self.documents:
            words = self.analyze_fn(document)
            i = 0
            while i < len(words):
                yield words[i:(i + self.max_sentence_length)]
                i += self.max_sentence_length


class Word2VecRetrieval(RetrievalBase, RetriEvalMixIn, CombinatorMixIn):
    """ Kwargs are passed down to RetrievalBase's countvectorizer,
    whose analyzer is then used to decompose the documents into tokens
    model - the Word Embedding model to use
    name  - identifier for the retrieval model
    verbose - verbosity level
    oov    - token to use for out of vocabulary words
    vocab_analyzer - analyzer to use to prepare for vocabulary filtering
    try_lowercase - try to match with an uncased word if cased failed
    >>> docs = ["the quick", "brown fox", "jumps over", "the lazy dog", "This is a document about coookies and cream and fox and dog", "The master thesis on the information retrieval task"]
    >>> sentences = StringSentence(docs)
    >>> from gensim.models import Word2Vec
    >>> model = Word2Vec(sentences, min_count=1)
    >>> word2vec = Word2VecRetrieval(model)
    >>> _ = word2vec.fit(docs)
    >>> values = word2vec.evaluate([(0,"fox"), (1,"dog")], [[0,1,0,0,1,0],[0,0,0,1,1,0]])
    >>> values['mean_average_precision']
    1.0
    >>> values['mean_reciprocal_rank']
    1.0
    >>> values['ndcg@k']
    (1.0, 0.0)
    """
    def __init__(self,
                 model,
                 name=None,
                 wmd=1.0,
                 verbose=0,
                 vocab_analyzer=None,
                 oov=None,
                 try_lowercase=False,
                 **matching_params):
        self.model = model
        self.wmd = wmd
        self.verbose = verbose
        self.try_lowercase = try_lowercase
        # inits self._cv
        if name is None:
            if not self.wmd:
                name = "wcd"
            else:
                name = "wcd+wmd"
        self._init_params(name=name, **matching_params)
        # uses cv's analyzer which can be specified by kwargs
        if vocab_analyzer is None:
            # if none, infer from analyzer used in matching
            self.analyzer = self._cv.build_analyzer()
        else:
            self.analyzer = vocab_analyzer
        self.oov = oov

    def _filter_oov_token(self, words):
        return [word for word in words if word != self.oov]

    def _medoid_expansion(self, words, n_expansions=1):
        """
        >>> from gensim.models import Word2Vec
        >>> model = Word2Vec(["brown fox".split()],min_count=1)
        >>> rtrvl = Word2VecRetrieval(model)
        >>> rtrvl._medoid_expansion(["brown"], n_expansions=1)
        ['brown', 'fox']
        """
        if n_expansions < 1:
            return words
        exps = self.model.most_similar(positive=words)[:n_expansions]
        exps, _scores = zip(*exps)
        exps = list(exps)
        if self.verbose > 0:
            print("Expanded", words, "by:", exps)
        return words + exps

    def fit(self, docs, y=None):
        self._fit(docs, y)
        # self._X = np.apply_along_axis(lambda d: self.analyzer(str(d)), 0, X)
        # sentences = [self.analyzer(doc) for doc in docs]
        # self.bigrams = Phrases(sentences)
        # sentences = [self.bigrams[sentence] for sentence in sentences]
        analyzed_docs = (self.analyzer(doc) for doc in docs)
        X = [filter_vocab(self.model, d, oov=self.oov) for d in analyzed_docs]

        self._X = np.asarray(X)
        return self

    def partial_fit(self, docs, y=None):
        self._partial_fit(docs, y)

        analyzed_docs = (self.analyzer(doc) for doc in docs)
        Xprep = np.asarray(
            [filter_vocab(self.model, doc, oov=self.oov) for doc in
                analyzed_docs]
        )
        self._X = np.hstack([self._X, Xprep])

    def query(self, query):
        model = self.model
        verbose = self.verbose
        indices = self._matching(query)
        wmd = self.wmd
        docs, labels = self._X[indices], self._y[indices]
        if verbose > 0:
            print(len(docs), "documents matched.")

        # if self.wmd:
        #     if self.wmd is True: wmd = k
        #     elif isinstance(self.wmd, int):
        #         wmd = k + self.wmd
        #     elif isinstance(self.wmd, float):
        #         wmd = int(k * self.wmd)
        #     else:
        #         raise ValueError("wmd= what?")
        # else:
        #     wmd = False

        q = self.analyzer(query)
        # q = self.bigrams[q]
        q = filter_vocab(self.model, q, oov=self.oov)

        # docs, labels set
        if verbose > 0:
            print("Preprocessed query:", q)
        if len(docs) == 0 or len(q) == 0:
            return []
        cosine_similarities = np.asarray(
            [model.n_similarity(q, doc) for doc in docs]
        )

        # nav
        # topk = argtopk(cosine_similarities, 1000, sort=not wmd)  # sort when wcd
        # # # It is important to also clip the labels #
        # docs, labels = docs[topk], labels[topk]
        # may be fewer than k

        ind = np.argsort(cosine_similarities)[::-1]
        if verbose > 0:
            print(cosine_similarities[ind])

        if not wmd:
            # no wmd, were done
            result = labels[ind]
        else:  # wmd TODO prefetch and prune
            # scores = np.asarray([model.wmdistance(self._filter_oov_token(q),
            #                                       self._filter_oov_token(doc))
            scores = np.asarray([model.wmdistance(q, doc)
                                 for doc in docs])
            ind = np.argsort(scores)
            if verbose > 0:
                print(scores[ind])
            result = labels[ind]

        # if not wmd:  # if wmd is False
        #     result = labels[:k]
        # else:
        #     if verbose > 0:
        #         print("Computing wmdistance")
        #     scores = np.asarray([model.wmdistance(self._filter_oov_token(q),
        #                                           self._filter_oov_token(doc))
        #                          for doc in docs])
        #     ind = np.argsort(scores)  # ascending by distance
        #     if verbose > 0:
        #         print(scores[ind])
        #     ind = ind[:k]             # may be more than k
        #     result = labels[ind]

        return result


class WordCentroidRetrieval(RetrievalBase, RetriEvalMixIn):
    def __init__(self, embedding, name="trueWCD", n_jobs=1, normalize=False, verbose=0, **kwargs):
        self.embedding = embedding
        self.normalize = normalize
        self.verbose = verbose
        self._init_params(**kwargs)  # initializes self._cv
        self.analyzer = self._cv.build_analyzer()

    def _compute_centroid(self, words):
        E = self.embedding
        centroid = np.mean(np.asarray([E[word] for word in words]),
                           axis=0)
        if self.verbose:
            print("Centroid shape:", centroid.shape)
        return centroid

    def fit(self, docs, y=None):
        analyze = self.analyzer
        self._fit(docs, y)  # we actually do not need the matching part
        centroids = np.asarray([self._compute_centroid(analyze(doc)) for doc in docs])
        if self.normalize:
            normalize(centroids, norm='l2', copy=False)

        self.nn = NearestNeighbors(n_jobs=self.n_jobs).fit(centroids)
        return self

    def query(self, query, k=None):
        # note that k is unused to compute recall properly
        if k is None:
            k = self.n_docs
        labels = self._y  # consider all documents
        # ind = self.matching(query)
        # centroids, labels = self.centroids[ind], self._y[ind]
        q_centroid = np.asarray([self._compute_centroid(self.analyzer(query))]).reshape(1, -1)
        if self.normalize:
            normalize(q_centroid, norm='l2', copy=False)
        if self.verbose > 0:
            print("Centered normalized query shape", q_centroid.shape)
        # sims = [cosine(q_centroid, centroid) for centroid in centroids]
        # ranks = np.argsort(sims)

        ind = self.nn.kneighbors(q_centroid, k, return_distance=False)[0]  # single query instance
        return labels[ind]


class WordMoversRetrieval(RetrievalBase, RetriEvalMixIn):
    """Retrieval based on the Word Mover's Distance"""
    def __init__(self, embedding, name="neoWMD", n_jobs=1, **kwargs):
        """initalize parameters"""
        pass


if __name__ == '__main__':
    import doctest
    doctest.testmod()
