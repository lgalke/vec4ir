from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import numpy as np


def uptrain(corpus,
            model_path=None,
            binary=True,
            lockf=0.0,
            min_count=1,
            size=300,
            **word2vec_params):
    wv = Word2Vec(min_count=min_count, size=size, **word2vec_params)
    print("Building vocabulary...")
    wv.build_vocab(corpus)
    print("Found %d distinct words." % len(wv.index2word))
    if model_path is not None:
        print("Intersecting with", model_path, "...")
        wv.intersect_word2vec_format(model_path, binary=binary, lockf=lockf)
        print("Intersected vectors locked with", lockf)

    total_examples = len(corpus)
    print("Training on %d documents..." % total_examples)
    wv.train(corpus, total_examples=total_examples)

    return wv


def all_but_the_top(v, D):
    """
    All-but-the-Top: Simple and Effective Postprocessing for Word
    Representations
    https://arxiv.org/abs/1702.01417

    Arguments:
        :v: word vectors of shape (n_words, n_dimensions)
        :D: number of principal components to subtract

    """
    print("All but the top")
    # 1. Compute the mean for v
    mu = np.mean(v, axis=0)
    v_tilde = v - mu  # broadcast hopefully works

    # 2. Compute the PCA components

    pca = PCA(n_components=D)
    u = pca.fit_transform(v.T)

    # 3. Postprocess the representations
    for w in range(v_tilde.shape[0]):
        v_tilde[w, :] -= np.sum([(u[:, i] * v[w]) * u[:, i].T for i in
                                 range(D)],
                                axis=0)

    return v_tilde
