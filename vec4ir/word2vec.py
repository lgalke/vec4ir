from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import sys
import logging
try:
    from .base import RetrievalBase, RetriEvalMixin
except SystemError:
    from base import RetrievalBase, RetriEvalMixin

default_analyzer = CountVectorizer().build_analyzer()


class StringSentence(object):
    """
    Classic approach to decompose documents into smaller word lists using a
    window_size argument
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


def argtopk(A, k, axis=-1):
    """
    >>> A = np.asarray([5,4,3,6,7,8,9,0])
    >>> argtopk(A, 3)
    array([6, 5, 4])
    >>> argtopk(A, 1)
    array([6])
    >>> argtopk(A, 0)
    """
    if k <= 0:
        raise UserWarning("k <= 0 in argtopk, result may be undesired")
    ind = np.argpartition(A, -k, axis=axis)[-k:]
    ind = ind[np.argsort(A[ind])][::-1]
    return ind



class Word2VecRetrieval(RetrievalBase, RetriEvalMixin):
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
    {'average_ndcg_at_k': 1.0,
     'mean_average_precision': 1.0,
     'mean_reciprocal_rank': 1.0}
    """
    def __init__(self, model, method="n_similarity", analyzer=None):
        if method not in ["n_similarity", "wmdistance"]:
            raise ValueError
        self.model = model
        self.method = method
        self._init_params(name='+'.join(["word2vec", method]))
        if analyzer is not None:
            self.analyzer = analyzer
        else:
            self.analyzer = default_analyzer

    def fit(self, docs, y=None):
        self._fit(docs, y)
        # self._X = np.apply_along_axis(lambda d: self.analyzer(str(d)), 0, X)
        self._X = np.asarray([self.analyzer(doc) for doc in docs])
        return self

    def partial_fit(self, docs, y=None):
        self._partial_fit(docs, y)
        Xprep = np.asarray([self.analyzer(doc) for doc in docs])
        self._X = np.hstack[self._X, Xprep]

    def query(self, query, k=1, verbose=0):
        model = self.model
        indices = self._matching(query, return_indices=True)
        docs, labels = self._X[indices], self._y[indices]
        if verbose > 0:
            print(len(docs), "documents matched.")
        n_ret = min(len(docs), k)
        if n_ret == 0:
            if verbose > 0:
                print("WARNING: NO MATCH")
            return []
        q = self.analyzer(query)
        if verbose > 0:
            print("analyzed:", q)
        q = [w for w in q if w in self.model]
        if verbose > 0:
            print("oov removed:", q)
            if len(q) == 0:
                print("WARNING: EMPTY QUERY")
                return []
        cosine_similarities = np.asarray([model.n_similarity(q, doc) for doc in docs])
        topk = argtopk(cosine_similarities, n_ret)
        ### It is important to also clip the labels
        docs, labels = docs[topk], labels[topk]
        if self.method == 'n_similarity':
            return labels
        elif self.method == 'wmdistance':
            scores = np.asarray([model.wmdistance(q, doc) for doc in docs])
            ind = np.argsort(scores)  # ascending
            return labels[ind]

if __name__ == '__main__':
    import doctest
    doctest.testmod()
