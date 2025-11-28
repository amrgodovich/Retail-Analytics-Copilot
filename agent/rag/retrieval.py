import os
import re
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """
      1. loading docs
      2. Splitting into chunks
      3. indexing using BM25
      4. returning top-k chunks
    """

    def __init__(self, docs_folder="docs"):
        self.docs_folder = docs_folder #path
        self.chunks = [] #chunks
        self.tokenized_chunks = [] #after tokenizing
        self.bm25 = None #bm25 model

        self.load_and_chunk()
        self.build_bm25()

    # loading and splitting
    def load_and_chunk(self):
        chunk_id = 0

        for filename in os.listdir(self.docs_folder):
            if not filename.endswith(".md"):
                continue

            path = os.path.join(self.docs_folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            # split as paragraphs 
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

            # print(f"Loaded {len(paragraphs)} paragraphs from {filename}")

            for i, part in enumerate(paragraphs):
                self.chunks.append({
                    "id": f"chunk{chunk_id}",
                    "filename": filename.replace(".md", ""),
                    "text": part
                })
                chunk_id += 1
            
            # print("first chunk:", self.chunks[-1])

    # Tokeninzing & BM25 indexing
    def build_bm25(self):
        self.tokenized_chunks = [chunk["text"].lower().split() for chunk in self.chunks]
        if self.tokenized_chunks:
            self.bm25 = BM25Okapi(self.tokenized_chunks)

    # retr chunks
    def retrieve(self, query: str, k: int = 5):
        """
            id, content, source (filename), score.
        """
        if not self.bm25:
            return []

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        # get top-k indexes sorted by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "id": chunk["id"],
                "filename": chunk["filename"],
                "score": float(scores[idx]),
                "text": chunk["text"]
            })

        return results

mytest = BM25Retriever()
print(mytest.retrieve("AOV", k=3))