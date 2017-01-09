# from gensim.models import Doc2Vec
from .base import RetrievalBase, RetriEvalMixIn
# from .word2vec import filter_vocab
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Doc2VecRetrieval(RetrievalBase, RetriEvalMixIn):
    def __init__(self,
                 intersect=None,
                 vocab_analyzer=None,
                 name=None,
                 verbose=0,
                 oov=None,
                 **kwargs):
        # self.model = model
        self.verbose = verbose
        self.oov = oov
        self.intersect = intersect
        if name is None:
            name = "paragraph-vectors"

        self._init_params(name=name, **kwargs)
        if vocab_analyzer is not None:
            self.analyzer = vocab_analyzer
        else:
            #  use analyzer of matching
            self.analyzer = self._cv.build_analyzer()

    def fit(self, docs, y):
        self._fit(docs, y)
        assert len(docs) == len(y)
        X = [TaggedDocument(self.analyzer(doc),
                            [label])
             for doc, label in zip(docs, y)]

        if self.verbose > 0:
            print("First 3 tagged documents:\n", X[:3])
            print("Training doc2vec model")
        # d2v = Doc2Vec()
        # d2v.build_vocab(X)
        # if self.intersect is not None:
        #     d2v.intersect_word2vec_format(self.intersect)
        self.model = Doc2Vec(X,
                            dm=1,
                            size=100,
                            window=8,
                            # alpha=15,
                            min_count=1,
                            sample=1e-5,
                            workers=16,
                            negative=20,
                            iter=20,
                            dm_mean=0,
                            dm_concat=1,
                            # dbow_words=1,
                            dm_tag_count=1)

        if self.verbose > 0:
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
            dv = model.docvecs[str(d)].reshape(1, -1)
            # sim = model.similarity(qv, dv)
            sim = cosine_similarity(qv, dv)[0]
            similarities.append(sim)

        similarities = np.asarray(similarities).reshape(1, -1)
        # similarities = [model.similarity(d, model.infer_vector(q)) for d in
        #                 docs]
        ind = np.argsort(similarities)[::-1]  # REVERSE! we want similar ones
        y = labels[ind]
        return y[0]
