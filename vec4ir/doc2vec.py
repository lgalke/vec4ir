from gensim.models import Doc2Vec
try:
    from .base import RetrievalBase, RetriEvalMixIn
except SystemError:
    from base import RetrievalBase, RetriEvalMixIn
# from .word2vec import filter_vocab
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Doc2VecRetrieval(RetrievalBase, RetriEvalMixIn):
    def __init__(self,
                 intersect=None,
                 vocab_analyzer=None,
                 name=None,
                 verbose=0,
                 n_epochs=10,
                 oov=None,
                 alpha=0.025,
                 min_alpha=0.005,
                 **kwargs):
        # self.model = model
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.verbose = verbose
        self.oov = oov
        self.intersect = intersect
        if name is None:
            name = "paragraph-vectors"

        self._init_params(name=name, **kwargs)
        self.analyzer = self._cv.build_analyzer()
        self.model = Doc2Vec(alpha=alpha,
                             min_alpha=alpha,
                             size=200,
                             window=8,
                             min_count=1,
                             sample=1e-5,
                             workers=8,
                             negative=20,
                             dm_mean=0,
                             dm_concat=1,
                             dm_tag_count=1
                             )
        self.n_epochs = n_epochs

    def fit(self, docs, y):
        self._fit(docs, y)
        assert len(docs) == len(y)
        model = self.model
        n_epochs = self.n_epochs
        verbose = self.verbose
        decay = (self.alpha - self.min_alpha) / n_epochs
        X = [TaggedDocument(self.analyzer(doc), [label])
             for doc, label in zip(docs, y)]

        if verbose > 0:
            print("First 3 tagged documents:\n", X[:3])
            print("Training doc2vec model")
        # d2v = Doc2Vec()
        # d2v.build_vocab(X)
        # if self.intersect is not None:
        #     d2v.intersect_word2vec_format(self.intersect)
        model.build_vocab(X)
        for epoch in range(n_epochs):
            if verbose:
                print("Doc2Vec: Epoch {} of {}.".format(epoch + 1, n_epochs))
            model.train(X)
            model.alpha -= decay  # apply global decay
            model.min_alpha = model.alpha  # but no decay inside one epoch

        if verbose > 0:
            print("Finished.")
            print("model:", self.model)

        return self

    def query(self, query):
        """ k unused """
        model = self.model
        verbose = self.verbose
        indices = self._matching(query)
        # docs, labels = self._X[indices], self._y[indices]
        labels = self._y[indices]
        if verbose > 0:
            print(len(labels), "documents matched.")
        q = self.analyzer(query)
        qv = model.infer_vector(q).reshape(1, -1)
        # similarities = [model.docvecs.similarity(model.docvecs[d],qv) for d in iter(labels)]
        similarities = []
        for d in labels:
            dv = model.docvecs[d].reshape(1, -1)
            # sim = model.similarity(qv, dv)
            sim = cosine_similarity(qv, dv)[0]
            similarities.append(sim)

        similarities = np.asarray(similarities).reshape(1, -1)
        # similarities = [model.similarity(d, model.infer_vector(q)) for d in
        #                 docs]
        ind = np.argsort(similarities)[::-1]  # REVERSE! we want similar ones
        y = labels[ind]
        return y[0]
