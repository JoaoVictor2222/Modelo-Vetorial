import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer, PorterStemmer


class Preprocessor:
    def __init__(self, language='portuguese'):
        self.language = language
        self._ensure_nltk_data()
        self.stop_words = set(stopwords.words(language))
        if language == 'portuguese':
            self.stemmer = RSLPStemmer()
        else:
            self.stemmer = PorterStemmer()

    def _ensure_nltk_data(self):
        for resource in ('corpora/stopwords',):
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download('stopwords', quiet=True)
                break
        if self.language == 'portuguese':
            try:
                nltk.data.find('stemmers/rslp')
            except LookupError:
                nltk.download('rslp', quiet=True)

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stop_words]

    def stem(self, tokens):
        return [self.stemmer.stem(t) for t in tokens]

    def process(self, text):
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem(tokens)
        return tokens
