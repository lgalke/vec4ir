#!/usr/bin/env python3
from retrieval import RetrievalModel, TermMatch
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
    QUERYTERMS = ["fox quick", "information retrieval"]
    for results in tfidf.query(QUERYTERMS):
        print(*[corpus[result] for result in results], sep='\n')

    print("Testing RetrievalModel with document ids...")
    X, y = zip(*[(doc, docid) for docid, doc in enumerate(corpus)])
    tfidf.fit(X, y)
    for results in tfidf.query(QUERYTERMS):
        print(*results, sep='\n')
