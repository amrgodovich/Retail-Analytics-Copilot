import os
import re
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

class BM25Retriever:
    """
      1. loading docs
      2. Splitting into chunks
      3. tokenizing and indexing using BM25
      4. returning top-k chunks
    """

    def __init__(self, docs_folder="docs"):
        self.docs_folder = docs_folder #path
        self.chunks = [] #chunks
        self.tokenized_chunks = [] #after tokenizing
        self.bm25 = None #bm25 model

        self.load_and_chunk()
        self.build_bm25()

    # preprocessing
    def preprocess(self, text):
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [stemmer.stem(token) for token in tokens]
        
        return tokens

    
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
        self.tokenized_chunks = [self.preprocess(chunk["text"]) for chunk in self.chunks]
        if self.tokenized_chunks:
            self.bm25 = BM25Okapi(self.tokenized_chunks)

    # retr chunks
    def retrieve(self, query: str, k: int = 5):
        """
            id, content, source (filename), score.
        """
        query_tokens = self.preprocess(query) 
        # print("Query tokens:", query_tokens)
        scores = self.bm25.get_scores(query_tokens)
        # print("Scores:", scores)

        score_index_pairs = []
        for i in range(len(scores)):
            score_index_pairs.append([scores[i], i])

        score_index_pairs.sort(reverse=True)
        top_related_indices = [pair[1] for pair in score_index_pairs[:k]]
        print("Top related indices:", top_related_indices)

        results = []
        for idx in top_related_indices:
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

# mytest = BM25Retriever()
# print(mytest.retrieve("According to the product policy, what is the return window (days) for unopened Beverages?"))