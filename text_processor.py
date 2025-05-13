from collections import Counter
from math import log

import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('all')

class TextProcessor:
    def __init__(self):
        self.stopwords = set(
            nltk.corpus.stopwords.words("russian") + nltk.corpus.stopwords.words("english")
        )
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        """Токенизация, удаление стоп-слов и лемматизация."""
        # Удаляем HTML-теги
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()

        # Приводим текст к нижнему регистру
        text = text.lower()

        # Токенизируем текст
        words = word_tokenize(text)

        # Удаляем пунктуацию и стоп-слова
        words = [
            word for word in words
            if word.isalpha() and word not in self.stopwords and word not in string.punctuation
        ]

        # Лемматизация
        words = [self.lemmatizer.lemmatize(word) for word in words]

        return " ".join(words)

    def calculate_tf_idf(self, texts):
        """Вычисление TF-IDF для списка текстов."""
        word_counts = [Counter(text.split()) for text in texts]
        total_documents = len(texts)

        # Словарь для хранения IDF
        idf = {}
        all_words = set(word for text in texts for word in text.split())
        for word in all_words:
            doc_count = sum(1 for wc in word_counts if word in wc)
            idf[word] = log(total_documents / (1 + doc_count)) + 1  # Корректировка IDF

        # Вычисляем TF-IDF для каждого слова
        tf_idf_results = []
        for wc in word_counts:
            total_words = sum(wc.values())
            for word, count in wc.items():
                tf = count / total_words
                tf_idf_results.append(
                    {
                        "word": word,
                        "tf": tf,
                        "idf": idf[word],
                        "tf_idf": tf * idf[word]
                    }
                )

        # Сортируем по убыванию TF-IDF
        return sorted(tf_idf_results, key=lambda x: x["tf_idf"], reverse=True)