import numpy as np
import keras.ops as ops


class KerasEnterpriseDataset:
    def __init__(self, json_path="mock_docs.json", batch_size=32):
        import json
        import math

        with open(json_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        self.batch_size = batch_size

        self.vocabulary = set()
        for doc in self.documents:
            words = self._tokenize(
                doc["content"] + " " + doc["title"] + " " + doc["id"]
            )
            self.vocabulary.update(words)
        self.vocab_list = sorted(list(self.vocabulary))
        self.vocab_index = {word: i for i, word in enumerate(self.vocab_list)}

        num_docs = len(self.documents)
        self.idf = np.zeros(len(self.vocab_list), dtype=np.float32)

        doc_counts = np.zeros(len(self.vocab_list), dtype=np.float32)
        for doc in self.documents:
            words = set(
                self._tokenize(doc["content"] + " " + doc["title"] + " " + doc["id"])
            )
            for word in words:
                if word in self.vocab_index:
                    doc_counts[self.vocab_index[word]] += 1.0

        for word, idx in self.vocab_index.items():
            self.idf[idx] = math.log((1.0 + num_docs) / (1.0 + doc_counts[idx])) + 1.0

    def _tokenize(self, text):
        """Izolează textul tehnic fără a sparge ID-urile care conțin underscore."""
        # NU eliminăm caracterul "_" pentru a proteja id-urile de tipul keras_rule_11
        for char in [
            "( ",
            " )",
            "(",
            ")",
            ".",
            "[",
            "]",
            "{",
            "}",
            ",",
            ";",
            '"',
            "'",
            "?",
            "!",
            "-",
        ]:
            text = text.replace(char, " ")
        return [word.strip().lower() for word in text.split() if len(word.strip()) > 1]

    def _text_to_vector(self, text):
        vector = np.zeros(len(self.vocab_list), dtype=np.float32)
        words = self._tokenize(text)
        for word in words:
            if word in self.vocab_index:
                vector[self.vocab_index[word]] += 1.0
        vector = vector * self.idf
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def get_all_vectors_optimized(self):
        num_docs = len(self.documents)
        vocab_size = len(self.vocab_list)
        all_vectors = np.zeros((num_docs, vocab_size), dtype=np.float32)

        for idx, doc in enumerate(self.documents):
            words = self._tokenize(doc["content"])
            title_words = self._tokenize(doc["title"])
            id_words = self._tokenize(doc["id"])

            for word in words:
                if word in self.vocab_index:
                    all_vectors[idx, self.vocab_index[word]] += 1.0
            for word in title_words:
                if word in self.vocab_index:
                    all_vectors[idx, self.vocab_index[word]] += 2.0
            for word in id_words:
                if word in self.vocab_index:
                    all_vectors[idx, self.vocab_index[word]] += (
                        5.0  # Boost masiv pentru potrivire pe ID
                    )

        all_vectors = all_vectors * self.idf
        norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
        all_vectors = np.where(norms > 0, all_vectors / norms, all_vectors)
        return all_vectors


def compute_keras_similarity(query_vector, all_vectors):
    all_vectors_tensor = ops.convert_to_tensor(all_vectors, dtype="float32")
    query_tensor = ops.convert_to_tensor(query_vector, dtype="float32")
    scores = ops.matmul(all_vectors_tensor, ops.expand_dims(query_tensor, axis=-1))
    return ops.squeeze(scores, axis=-1)
