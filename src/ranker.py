import math


class Ranking:
    def __init__(self):
        self.results = []

    def cosine_similarity(self, vec1, vec2):
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0

        for term, weight in vec1.items():
            norm1 += weight ** 2
            if term in vec2:
                dot_product += weight * vec2[term]

        for weight in vec2.values():
            norm2 += weight ** 2

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))

    def rank(self, query_vector, documents):
        self.results = []
        for doc in documents:
            score = self.cosine_similarity(query_vector, doc.vector)
            if score > 0:
                self.results.append((doc.doc_id, doc.filename, score))

        self.results.sort(key=lambda x: x[2], reverse=True)
        return self.results

    def display(self, top_n=10):
        if not self.results:
            print("Nenhum documento relevante encontrado.")
            return

        print(f"\n{'Posicao':<8} {'Documento':<30} {'Similaridade':<12}")
        print("-" * 50)
        for i, (doc_id, filename, score) in enumerate(self.results[:top_n], 1):
            print(f"{i:<8} {filename:<30} {score:<12.6f}")
