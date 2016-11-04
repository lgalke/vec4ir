from gensim.models import Word2Vec
from base import RetrievalBase, RetriEvalMixin
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import sys
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


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
    >>> sentences = StringSentence(analyze_fn, documents, 3)
    >>> x = list(sentences)
    >>> len(x)
    3
    >>> x[2]
    ['the', 'lazy', 'dog']
    >>> sentences = StringSentence(analyze_fn, documents, 5)
    >>> x = list(sentences)
    >>> len(x)
    2
    >>> x[0]
    ['the', 'quick', 'brown', 'fox', 'jumps']
    >>> x[1]
    ['over', 'the', 'lazy', 'dog']
    """

    def __init__(self, analyze_fn, documents, max_sentence_length=10000):
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

default_analyzer = CountVectorizer().build_analyzer()


class Word2VecRetrieval(RetrievalBase, RetriEvalMixin):
    """ Kwargs are passed down to RetrievalBase's countvectorizer,
    whose analyzer is then used to decompose the documents into tokens
    >>> docs = ["the quick", "brown fox", "jumps over", "the lazy dog", "This is a document about coookies and cream and fox and dog", "why did you chose to do a masters thesis on the information retrieval task"]
    >>> sentences = StringSentence(default_analyzer, docs)
    >>> model = Word2Vec(sentences, min_count=1)
    >>> word2vec = Word2VecRetrieval(model)
    >>> _ = word2vec.fit(docs)
    >>> values = word2vec.score(["fox", "dog"], [[0,1,0,0,1,0],[0,0,0,1,1,0]])
    >>> import pprint
    >>> pprint.pprint(values)
    {'average_ndcg_at_k': 1.0,
     'mean_average_precision': 1.0,
     'mean_reciprocal_rank': 1.0}
    """
    def __init__(self, model, analyzer=None):
        self.model = model
        self._init_params()  # also inits _cv
        self.analyzer = self._cv.build_analyzer() if analyzer is None else analyzer

    def fit(self, X, y=None):
        self._fit(X, y)
        return self

    def query(self, X, k=1):
        model = self.model
        for q in X:
            docs, ids = self._matching(q)
            print(q, file=sys.stderr)
            n_ret = min(len(docs), k)
            q = self.analyzer(q)
            scores = np.apply_along_axis(
                lambda d:
                model.wmdistance(q, self.analyzer(str(d))), 0, docs)
            ind = np.argsort(scores)[:n_ret+1]
            labels = ids[ind]
            print(labels, file=sys.stderr)
            yield labels

if __name__ == '__main__':
    import doctest
    doctest.testmod()
