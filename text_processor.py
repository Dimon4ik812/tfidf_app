import os
import nltk
from math import log
from collections import Counter

# Скачиваем стоп-слова NLTK
if not os.path.exists(nltk.data.find("corpora/stopwords")):
    nltk.download("stopwords")
STOPWORDS = set(
    nltk.corpus.stopwords.words("russian") + nltk.corpus.stopwords.words("english")
)


class TextProcessor:
    def __init__(self):
        self.stopwords = STOPWORDS

    def preprocess_text(self, text):
        """Токенизация и удаление стоп-слов."""
        words = text.lower().split()
        return " ".join([word for word in words if word not in self.stopwords])

    def calculate_tf_idf(self, texts):
        """Вычисление TF-IDF для списка текстов."""
        word_counts = [Counter(text.split()) for text in texts]
        total_documents = len(texts)

        # Словарь для хранения IDF
        idf = {}
        for word in set(word for text in texts for word in text.split()):
            doc_count = sum(1 for wc in word_counts if word in wc)
            idf[word] = log(total_documents / (1 + doc_count))

        # Вычисляем TF-IDF для каждого слова
        tf_idf_results = []
        for wc in word_counts:
            total_words = sum(wc.values())
            for word, count in wc.items():
                tf = count / total_words
                tf_idf_results.append(
                    {"word": word, "tf": tf, "idf": idf[word], "tf_idf": tf * idf[word]}
                )

        return sorted(tf_idf_results, key=lambda x: x["idf"], reverse=True)
