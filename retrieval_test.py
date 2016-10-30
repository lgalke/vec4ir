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
    print(tfidf)
    QUERYTERMS = ["fox quick", "information retrieval"]
    print("Testing RetrievalModel with document ids...")
    X, y = zip(*[(doc, docid) for docid, doc in enumerate(corpus)])
    tfidf.fit(X, y)
    for query, result in zip(QUERYTERMS,tfidf.query(QUERYTERMS, 3)):
        print("QUERY:",query)
        print(result)
        for docid in result:
            print(docid,corpus[int(docid)])
    print("TESTING SCORING!!!")
    import numpy as np
    rels = np.array([[0,1,0,1],[1,0,0,0]])
    print(rels.shape)
    scores = tfidf.score(QUERYTERMS, rels, k=20)
    pprint.pprint(scores)

