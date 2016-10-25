class InformationRetrieval:
    def __init__(self, similarity, analyzer, matcher=lambda x: True):
        self.matcher = matcher
        self.similarity = similarity
        self.analyzer = analyzer
        self.docs = []

    def index(self, new_docs):
        for doc in new_docs:
            self.docs.append(self.analyzer.analyze(doc))

    def query(self, query):
        analyzed_query = self.analyzer.analyze(query)
        matched = self.matcher.match(analyzed_query, self.docs)
        scores = self.similarity.score(matched)
        return zip(matched, scores)
