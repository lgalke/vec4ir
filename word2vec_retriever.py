from tflearn.layers.embedding_ops import embedding

    


class Word2vecRetriever(object):
    def __init__(self, preprocessor, vocab_size, embedding_size):
            
        self.preprocessor = preprocessor
