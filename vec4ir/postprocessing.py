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
      All-but-the-Top: Simple and Effective Postprocessing for Word Representations
      https://arxiv.org/abs/1702.01417
      Arguments:
          :v: word vectors of shape (n_words, n_dimensions)
          :D: number of principal components to subtract
      """
      # 1. Subtract mean vector
      v_tilde = v - np.mean(v, axis=0)
      # 2. Compute the first `D` principal components
      #    on centered embedding vectors
      u = PCA(n_components=D).fit(v_tilde).components_  # [D, emb_size]
      # Subtract first `D` principal components
      # [vocab_size, emb_size] @ [emb_size, D] @ [D, emb_size] -> [vocab_size, emb_size]
      return v - (v_tilde @ u.T @ u)  
