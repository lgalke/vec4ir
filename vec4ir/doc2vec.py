from gensim.models import Doc2Vec
try:
    from .base import RetrievalBase, RetriEvalMixin
except SystemError:
    from base import RetrievalBase, RetriEvalMixin
# from .word2vec import filter_vocab
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


RetriEvalMixin
