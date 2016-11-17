from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import sys
import logging
try:
    from .base import RetrievalBase, RetriEvalMixIn, CombinatorMixIn
except SystemError:
    from base import RetrievalBase, RetriEvalMixIn, CombinatorMixIn

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


def argtopk(A, k, axis=-1, sort=True):
    """
    >>> A = np.asarray([5,4,3,6,7,8,9,0])
    >>> argtopk(A, 3)
    array([6, 5, 4])
    >>> argtopk(A, 1)
    array([6])
    >>> argtopk(A, -3)
    array([7, 2, 1])
    >>> argtopk(A, -1)
    array([7])
    >>> argtopk(A, -6)
    array([7, 2, 1, 0, 3, 4])
    >>> argtopk(A, 6)
    array([6, 5, 4, 3, 0, 1])
    """
    if k == 0:
        raise UserWarning("k == 0? result [] may be undesired.")
        return []
    ind = np.argpartition(A, -k, axis=axis)
    ind = ind[-k:] if k > 0 else ind[:-k]

    if sort:
        ind = ind[np.argsort(A[ind])]
        if k > 0:
            ind = ind[::-1]

    return ind


class Word2VecRetrieval(RetrievalBase, RetriEvalMixIn, CombinatorMixIn):
    """ Kwargs are passed down to RetrievalBase's countvectorizer,
    whose analyzer is then used to decompose the documents into tokens
    >>> docs = ["the quick", "brown fox", "jumps over", "the lazy dog", "This is a document about coookies and cream and fox and dog", "why did you chose to do a masters thesis on the information retrieval task"]
    >>> sentences = StringSentence(docs)
    >>> model = Word2Vec(sentences, min_count=1)
    >>> word2vec = Word2VecRetrieval(model)
    >>> _ = word2vec.fit(docs)
    >>> values = word2vec.evaluate([(0,"fox"), (1,"dog")], [[0,1,0,0,1,0],[0,0,0,1,1,0]])
    >>> import pprint
    >>> pprint.pprint(values)
    {'mean_average_precision': 1.0,
     'mean_reciprocal_rank': 1.0,
     'ndcg_at_k': array([ 1.,  1.])}
    """
    def __init__(self, model, name=None, matching=True, method="wcd",
                 n_expansions=None, **kwargs):
        if method not in ["wcd", "wmd"]:
            raise ValueError
        self.model = model
        self.method = method
        self.n_expansions = n_expansions
        self.matching = matching
        self.verbose = kwargs.get('verbose', 0)
        # inits self._cv
        if name is None:
            name = '+'.join(["w2v", method])
        self._init_params(name=name, **kwargs)
        # uses cv's analyzer which can be specified by kwargs
        self.analyzer = self._cv.build_analyzer()

    def _filter_vocab(self, words, analyze=False):
        """ if analyze is given, analyze words first (split string) """
        if analyze:
            words = self.analyzer(words)
        filtered = [word for word in words if word in self.model]
        # if len(filtered) == 0:
        #     print("NO MATCH IN VOCAB:", words)
        return filtered

    def _medoid_expansion(self, words, n_expansions=1):
        """
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
            print("Expanded", words, "by:", exps, file=sys.stderr)
        return words + exps

    def fit(self, docs, y=None):
        self._fit(docs, y)
        # self._X = np.apply_along_axis(lambda d: self.analyzer(str(d)), 0, X)
        self._X = np.asarray(
            [self._filter_vocab(doc, analyze=True) for doc in docs]
        )
        return self

    def partial_fit(self, docs, y=None):
        self._partial_fit(docs, y)

        Xprep = np.asarray(
            [self._filter_vocab(doc, analyze=True) for doc in docs]
        )
        self._X = np.hstack([self._X, Xprep])

    def query(self, query, k=1):
        wcd = self.method == 'wcd'
        model = self.model
        q = self._filter_vocab(query, analyze=True)
        if len(q) > 0 and self.n_expansions:
            q = self._medoid_expansion(q, n_expansions=self.n_expansions)
        if self.matching:
            indices = self._matching(' '.join(q))
            docs, labels = self._X[indices], self._y[indices]
        else:
            docs, labels = self._X, self._y
        # docs, labels set
        n_ret = min(len(docs), k)
        if self.verbose > 0:
            print("preprocessed query:", q)
            print(len(docs), "documents matched.")
        if n_ret == 0 or len(q) == 0:
            return []
        cosine_similarities = np.asarray(
            [model.n_similarity(q, doc) if len(doc) > 0 else 0 for doc in docs]
        )
        topk = argtopk(cosine_similarities, n_ret, sort=wcd)  # sort when wcd
        # It is important to also clip the labels #
        docs, labels = docs[topk], labels[topk]
        if wcd:
            return labels
        scores = np.asarray([model.wmdistance(q, doc) for doc in docs])
        ind = np.argsort(scores)  # ascending
        ind = ind[:k]
        return labels[ind]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
