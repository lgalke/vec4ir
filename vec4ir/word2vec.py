from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import sys
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
    >>> A = [5,4,3,6,7,8,9,0]
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
    >>> argtopk(A, 10)
    array([6, 5, 4, 3, 0, 1, 2, 7])
    """
    k = min(len(A), k)
    A = np.asarray(A)
    if k == 0:
        raise UserWarning("k <= 0? result [] may be undesired.")
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
    model - the Word Embedding model to use
    name  - identifier for the retrieval model
    verbose - verbosity level
    oov    - token to use for out of vocabulary words
    vocab_analyzer - analyzer to use to prepare for vocabulary filtering
    try_lowercase - try to match with an uncased word if cased failed
    >>> docs = ["the quick", "brown fox", "jumps over", "the lazy dog", "This is a document about coookies and cream and fox and dog", "why did you chose to do a masters thesis on the information retrieval task"]
    >>> sentences = StringSentence(docs)
    >>> model = Word2Vec(sentences, min_count=1)
    >>> word2vec = Word2VecRetrieval(model)
    >>> _ = word2vec.fit(docs)
    >>> values = word2vec.evaluate([(0,"fox"), (1,"dog")], [[0,1,0,0,1,0],[0,0,0,1,1,0]])
    >>> values['mean_average_precision']
    1.0
    >>> values['mean_reciprocal_rank']
    1.0
    >>> values['ndcg_at_k']
    array([ 1.,  1.])
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

    def _filter_vocab(self, words, analyze=False):
        """ if analyze is given, analyze words first (split string) """
        if analyze:
            words = self.analyzer(words)
        # filtered = [word if word in self.model else self.oov for word in words]
        # TODO try lowercasing
        filtered = []
        for word in words:
            if word in self.model:
                filtered.append(word)
            elif self.try_lowercase and word.lower() in self.model:
                filtered.append(word.lower())  # FIXME could be optimized
                print("YAY hit a word with lowering!")
            elif self.oov is not None:
                filtered.append(self.oov)
        return filtered

    def _filter_oov_token(self, words):
        return [word for word in words if word != self.oov]

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
        # sentences = [self.analyzer(doc) for doc in docs]
        # self.bigrams = Phrases(sentences)
        # sentences = [self.bigrams[sentence] for sentence in sentences]
        X = [self._filter_vocab(doc, analyze=True) for doc in docs]

        self._X = np.asarray(X)
        return self

    def partial_fit(self, docs, y=None):
        self._partial_fit(docs, y)

        Xprep = np.asarray(
            [self._filter_vocab(doc, analyze=True) for doc in docs]
        )
        self._X = np.hstack([self._X, Xprep])

    def query(self, query, k=1, sort=True):
        model = self.model
        verbose = self.verbose
        indices = self._matching(query)
        docs, labels = self._X[indices], self._y[indices]
        if verbose > 0:
            print(len(docs), "documents matched.")

        if self.wmd:
            if self.wmd is True:
                wmd = k
            elif isinstance(self.wmd, int):
                wmd = k + self.wmd
            elif isinstance(self.wmd, float):
                wmd = int(k * self.wmd)
            else:
                raise ValueError("wmd= what?")
        else:
            wmd = False

        q = self.analyzer(query)
        # q = self.bigrams[q]
        q = self._filter_vocab(q)

        # docs, labels set
        if verbose > 0:
            print("Preprocessed query:", q)
        if len(docs) == 0 or len(q) == 0:
            return []
        cosine_similarities = np.asarray(
            [model.n_similarity(q, doc) for doc in docs]
        )

        topk = argtopk(cosine_similarities, k, sort=not wmd)  # sort when wcd
        # It is important to also clip the labels #
        docs, labels = docs[topk], labels[topk]
        # may be fewer than k

        if not wmd:  # if wmd is False
            result = labels[:k]
        else:
            if verbose > 0:
                print("Computing wmdistance")
            scores = np.asarray([model.wmdistance(q, doc) for doc in docs])
            ind = np.argsort(scores)  # ascending by distance
            if verbose > 0:
                print(scores[ind])
            ind = ind[:k]             # may be more than k
            result = labels[ind]

        return result


if __name__ == '__main__':
    import doctest
    doctest.testmod()
