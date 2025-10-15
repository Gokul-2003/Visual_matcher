import numpy as np
import json

class VisualMatcher:
    """
    Loads precomputed embeddings and metadata,
    and provides a query function to find top-k similar images.
    """
    def __init__(self, embeddings_path='embeddings.npy', metadata_path='metadata.json'):
        self.embeddings = np.load(embeddings_path)
        with open(metadata_path, 'r') as f:
            self.files = json.load(f)
        # normalize embeddings
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

    def query(self, q_emb, topk=5):
        q_emb = q_emb / np.linalg.norm(q_emb)
        sims = np.dot(self.embeddings, q_emb)
        idxs = sims.argsort()[::-1][:topk]
        results = []
        for i in idxs:
            results.append({'file': self.files[i], 'score': float(sims[i])})
        return results
