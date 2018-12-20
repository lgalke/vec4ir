import sys
from vec4ir import Retrieval, Matching, Tfidf
from vec4ir.query_expansion import CentroidExpansion, EmbeddedQueryExpansion
from gensim.models import Word2Vec

DOCUMENTS = ["The quick brown fox jumps over the lazy dog",
             "Surfing surfers do surf on green waves"]

def test_centroid_expansion():
    model = Word2Vec([doc.split() for doc in DOCUMENTS], iter=1, min_count=1)
    m = 2
    expansion = CentroidExpansion(model.wv, m=m)
    expansion.fit(DOCUMENTS)
    query = "surf"
    expanded_query = expansion.transform(query)
    # surf => surf surf Surfing
    print(query, expanded_query, sep='=>')
    assert len(expanded_query.split()) == len(query.split()) + m

def test_embedded_query_expansion():
    model = Word2Vec([doc.split() for doc in DOCUMENTS], iter=1, min_count=1)
    m = 2
    expansion = EmbeddedQueryExpansion(model.wv, m=m)
    expansion.fit(DOCUMENTS)
    query = "surf"
    expanded_query = expansion.transform(query)
    # surf => surf surf Surfing
    print(query, expanded_query, sep='=>')
    assert len(expanded_query.split()) == len(query.split()) + m

def test_expansion_inside_retrieval():
    # Integration test within full retrieval pipeline
    model = Word2Vec([doc.split() for doc in DOCUMENTS], iter=1, min_count=1)
    n_expansions = 2

    tfidf = Tfidf()
    match_op = Matching()
    expansion_op = EmbeddedQueryExpansion(model.wv, m=n_expansions)

    retrieval = Retrieval(tfidf,  # The retrieval model
                          matching=match_op,
                          query_expansion=expansion_op)
    ids = ['fox_ex', 'surf_ex']
    retrieval.fit(DOCUMENTS, ids)
    result = retrieval.query('surfing surfers do surf green')
    assert result[0] == 'surf_ex'


