#!/usr/bin/env python3
from retrieval import RetrievalModel, ExactMatch
if __name__ == '__main__':
    tfidf = RetrievalModel()
    corpus = [
        "Some information is in this document, let's see if it gets retrieved",
        "The quick brown fox jumps over the lazy dog"
    ]
    print("Testing RetrievalModel without document ids...")
    tfidf.fit(corpus)
    newdoc = "Additional information about fox"
    corpus.append(newdoc)
    print("Testing partial fit...")
    tfidf.partial_fit([newdoc])
    print(tfidf)
    QUERYTERMS = ["fox", "information retrieval"]
    for queryterm in QUERYTERMS:
        print("Querying for '{}'".format(queryterm))
        indices = tfidf.query([queryterm], 3)
        for index in indices[0]:
            print(corpus[index])

    print("Testing ExactMatch function")
    import numpy as np
    Xq = np.array([0, 1, 0])
    X = np.array([[1, 0, 3], [2, 3, 4]])
    indices = ExactMatch(Xq, X)
    print(indices)
    print(X[indices])

    print("Testing RetrievalModel with document ids...")
    X, y = zip(*[(doc, docid) for docid, doc in enumerate(corpus)])
    X = list(X)
    y = list(y)
    tfidf.fit(X, y)
    for queryterm in QUERYTERMS:
        print("Querying for '{}'".format(queryterm))
        print(tfidf.query([queryterm]))

    print("Testing querying for a list at once")
    print(tfidf.query(QUERYTERMS))

    # pipeline = TfIdfRetrieval()
    # pipeline.fit(corpus)
    # print(pipeline.kneighbors(["fox"], n_neighbors=3)[0])
