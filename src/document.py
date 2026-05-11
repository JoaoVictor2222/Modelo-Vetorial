class Document:
    def __init__(self, doc_id, filename, text):
        self.doc_id = doc_id
        self.filename = filename
        self.text = text
        self.tokens = []
        self.vector = {}

    def __repr__(self):
        return f"Document({self.doc_id}, {self.filename})"
