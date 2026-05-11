class Query:
    def __init__(self, text, vector=None):
        self.text = text
        self.tokens = []
        self.vector = vector or {}

    def __repr__(self):
        return f"Query({self.text})"
