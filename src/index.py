from collections import defaultdict


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)
        self.doc_freq = defaultdict(int)

    def add_document(self, doc_id, tokens):
        freq = defaultdict(int)
        for token in tokens:
            freq[token] += 1

        for term, count in freq.items():
            self.index[term].append((doc_id, count))
            self.doc_freq[term] += 1

    def get_postings(self, term):
        return self.index.get(term, [])

    def get_document_frequency(self, term):
        return self.doc_freq.get(term, 0)

    def get_terms(self):
        return list(self.index.keys())

    def __repr__(self):
        return f"InvertedIndex({len(self.index)} termos)"
