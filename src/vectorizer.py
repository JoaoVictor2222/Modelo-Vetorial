import math
from collections import defaultdict


class Vectorizer:
    def __init__(self, inverted_index, num_docs):
        self.index = inverted_index
        self.num_docs = num_docs

    def tf(self, freq):
        if freq == 0:
            return 0
        return 1 + math.log10(freq)

    def idf(self, term):
        df = self.index.get_document_frequency(term)
        if df == 0:
            return 0
        return 1 + math.log10(self.num_docs / df)

    def tfidf(self, term, freq):
        return self.tf(freq) * self.idf(term)

    def build_document_vector(self, doc_id, tokens):
        vector = {}
        freq = defaultdict(int)
        for token in tokens:
            freq[token] += 1

        for term, count in freq.items():
            vector[term] = self.tfidf(term, count)

        return vector

    def build_query_vector(self, query_tokens):
        vector = {}
        freq = defaultdict(int)
        for token in query_tokens:
            freq[token] += 1

        for term, count in freq.items():
            vector[term] = self.tfidf(term, count)

        return vector
