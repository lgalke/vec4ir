from gensim.models import Word2Vec


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
