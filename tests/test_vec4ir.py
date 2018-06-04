import os

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

from vec4ir import Doc2VecInference, Retrieval, Matching, Tfidf, WordCentroidDistance, build_analyzer, WordMoversDistance

documents = ["The quick brown fox jumps over the lazy dog",
             "Computer scientists are lazy lazy lazy"]

def test_build_analyzer():
    analyzer =build_analyzer('sklearn', False, lowercase=True)
    analyzed = analyzer("the quick brown fox")
    assert analyzed[-1] == "fox"


DEFAULT_ANALYZER = build_analyzer('sklearn', stop_words=False, lowercase=True)
TEST_FILE = "test.tmp"


def test_matching():
    match_op = Matching()
    match_op.fit(documents)
    matched = match_op.predict("fox")
    assert matched == [0]


def test_tfidf():
    tfidf = Tfidf()
    tfidf.fit(documents)
    result = tfidf.query('lazy')
    assert result[0] == 1
    assert result[1] == 0

def test_retrieval():
    tfidf = Tfidf()
    retrieval = Retrieval(tfidf)
    ids = ['fox_example', 'lazy_example']
    retrieval.fit(documents, ids)
    result = retrieval.query('fox')
    assert result[0] == 'fox_example'
    assert result[1] == 'lazy_example'

def test_word2vec():
    model = Word2Vec([doc.split() for doc in documents], iter=1, min_count=1)
    match_op = Matching()
    wcd = WordCentroidDistance(model.wv)
    retrieval = Retrieval(wcd, matching=match_op)
    retrieval.fit(documents)
    result = retrieval.query('dog')
    assert result[0] == 0

# PYEMD is required
# def test_wordmovers():
#     model = Word2Vec([doc.split() for doc in documents], iter=1, min_count=1)
#     match_op = Matching()
#     wmd = WordMoversDistance(model.wv)
#     retrieval = Retrieval(wmd, matching=match_op)
#     retrieval.fit(documents)
#     result = retrieval.query('dog')
#     assert result[0] == 0

def test_doc2vec_inference():
    tagged_docs = [TaggedDocument(simple_preprocess(doc), [i])
                   for i, doc in enumerate(documents)]
    model = Doc2Vec(tagged_docs, epochs=1, min_count=1)
    d2v = Doc2VecInference(model, DEFAULT_ANALYZER)
    match_op = Matching()
    retrieval = Retrieval(d2v, matching=match_op).fit(documents)
    result = retrieval.query("scientists")
    assert result[0] == 1

def test_doc2vec_inference_saveload():
    tagged_docs = [TaggedDocument(simple_preprocess(doc), [i])
                   for i, doc in enumerate(documents)]
    model = Doc2Vec(tagged_docs, epochs=1, min_count=1, vector_size=10)
    model.save(TEST_FILE)
    del model
    model = Doc2Vec.load(TEST_FILE)
    os.remove(TEST_FILE)
    d2v = Doc2VecInference(model, DEFAULT_ANALYZER)
    match_op = Matching()
    retrieval = Retrieval(d2v, matching=match_op).fit(documents)
    result = retrieval.query("scientists")
    assert result[0] == 1

