import os
import string

import nltk
import re

_alphabet = set(string.ascii_lowercase + string.digits + ' ')
word_regexp = r"(?u)\b[a-zA-Z_][a-zA-Z_]+\b"


class NltkNormalizer:
    def __init__(self):
        self.install_nltk_corpora('stopwords', 'wordnet', 'punkt')
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.lemmatizer.lemmatize('')  # Force nltk lazy corpus loader to do something.
        self.tokenizer = self.make_tokenizer()
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.sent_tokenizer = None


    @staticmethod
    def make_tokenizer():
        token_pattern = re.compile(word_regexp)
        return lambda doc: token_pattern.findall(doc)

    def split_and_normalize(self, o):
        r = []
        for t in self.tokenizer(o):
            if t not in self.stopwords and len(t) > 2:
                t = self.lemmatizer.lemmatize(t).lower()
                t = ''.join([lc for lc in t if lc in _alphabet])
                r.append(t)
        return r

    def normalize(self, o):
        return ' '.join(self.split_and_normalize(o))

    def sent_tokenize(self, doc):
        if not self.sent_tokenizer:
            self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return self.sent_tokenizer.tokenize(doc)

    @staticmethod
    def install_nltk_corpora(*packages):
        nltk_packages = list(packages)
        try:
            installed = (set(os.listdir(nltk.data.find("corpora"))) |
                         (set(os.listdir(nltk.data.find("taggers"))))) | \
                        (set(os.listdir(nltk.data.find("tokenizers"))))
        except LookupError:
            installed = set()
        if not set(nltk_packages) <= set(installed):
            nltk.download(nltk_packages)
