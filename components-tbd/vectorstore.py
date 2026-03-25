import faiss
import numpy as np

import os
import pickle

class FAISSStore:
    def __init__(self, dim, storage_dir="databases", name="faiss_index"):
        self.dim = dim
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, f"{name}.index")
        self.texts_path = os.path.join(storage_dir, f"{name}_texts.pkl")

        if os.path.exists(self.index_path) and os.path.exists(self.texts_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.texts_path, "rb") as f:
                self.texts = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.texts = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings).astype("float32"))
        self.texts.extend(texts)
        self.save()

    def search(self, query_embedding, k=5):
        D, I = self.index.search(
            np.array([query_embedding]).astype("float32"), k
        )
        return [self.texts[i] for i in I[0] if i < len(self.texts)]

    def save(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.texts_path, "wb") as f:
            pickle.dump(self.texts, f)

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.texts_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.texts_path, "rb") as f:
                self.texts = pickle.load(f)
        else:
            raise FileNotFoundError("No saved FAISS index or texts found.")