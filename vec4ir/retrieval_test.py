#!/usr/bin/env python3
from retrieval import RetrievalModel, TermMatch
import pprint
if __name__ == '__main__':
    tfidf = RetrievalModel()
    corpus = [
        "Some information is in this document, let's see if it gets retrieved",
        "The quick brown fox jumps over the lazy dog",
        "fox doc"
    ]
    print("Testing RetrievalModel without document ids...")
    tfidf.fit(corpus)
    newdoc = "Additional quick information about fox"
    corpus.append(newdoc)
    print("Testing partial fit...")
    tfidf.partial_fit([newdoc])
    QUERYTERMS = ["fox quick", "information retrieval"]
    print("Testing RetrievalModel with document ids...")
    # generate doc ids
    X, y = zip(*[(doc, docid) for docid, doc in enumerate(corpus)])
    import numpy as np
    tfidf.fit(X, y)
    for query, result in zip(QUERYTERMS, tfidf.query(QUERYTERMS, 3)):
        print("QUERY:", query)
        print(result)
        for docid in result:
            print(docid, corpus[int(docid)])
    print("TESTING SCORING!!!")
    print("... with nparray rels")
    rels = np.array([[0, 1, 0, 1], [1, 0, 0, 0]])
    scores = tfidf.score(QUERYTERMS, rels, k=20)
    pprint.pprint(scores)

    print("... with list of dict rels")
    reldict = [{0: 0, 1: 1, 2: 0, 3: 1}, {0: 1, 1: 0, 2: 0, 3: 0}]
    pprint.pprint(tfidf.score(QUERYTERMS, reldict, k=20))

